"""
Ruta del A* la cual actua como la curva central del ducto con radio de curvatura exacto, es decir >= R_min. Revision de
colisiones contra los bloques del entorno. Es el puente entre ruteo_astar.py de mi compañero Fer y el archivo ducto.py encargado del CAD.

A* de 26 vecinos devuelve una ruta en zigzag
Suavizarla con Chaikin no conoce R_min, debido a que el radio que sale depende del largo de los segmentos y del
paso de muestreo. En este codigo se hace al reves, primero se decide el radio y despues se construye la curva para cumplirlo.

Flujo:
  ruta del A*
    -> simplificar_ruta       "string pulling" quita vertices intermedios si la recta directa sigue libre de bloques
    -> redondear_esquinas     en cada vertice mete un arco circular tangente con un radio R
    -> holguras / revision    distancia real de la curva a cada bloque, menos el radio del ducto
  generar_ducto(..., curva=...)  hace el CAD con esa curva

En un vertice donde la ruta gira un angulo ϴ, un arco de radio R tangente a las dos rectas empieza a T = R*tan(theta/2) ANTES del vertice y
termina T despues. Para que quepa, T no puede pasar de lo que mide el segmento (se reparte la
mitad a cada vertice vecino). Si no cabe, el radio logrado es R' = T_disponible/tan(theta/2) < R y
se reporta la esquina como no factible de esta forma el modulo es transparente y avisa del problema

Distancia punto-caja a lo largo de un segmento es una funcion convexa, por eso su minimo se busca con busqueda ternaria y es exacto 

"""
from __future__ import annotations

import importlib
import math

import numpy as np

from cad.ducto import _EPS, _como_puntos, generar_ducto, validar_radio_minimo, validar_secciones_control

# Debe coincidir con ruteo_astar.DISTANCIA_MM_POR_UNIDAD.
_MM_POR_UNIDAD = {"termico": 1.0, "electrico": 1.0, "neutro": 0.0}


# Bloques a cajas

def _dseg_defecto(tipo, valor):
    return max(0.0, float(valor)) * _MM_POR_UNIDAD.get(tipo, 0.0)


def cajas_desde_bloques(bloques, distancia_seguridad=None) -> list[dict]:
    """Bloques con el formato {"nombre","centro":{x,y,z},"size":{x,y,z},"tipo","valor"} pasan a
    [{"nombre", "caja": (x0,y0,z0,x1,y1,z1), "dseg": mm}]."""
    dseg = distancia_seguridad or _dseg_defecto
    cajas = []
    for b in bloques:
        c, s = b["centro"], b["size"]
        caja = (
            c["x"] - s["x"] / 2.0, c["y"] - s["y"] / 2.0, c["z"] - s["z"] / 2.0,
            c["x"] + s["x"] / 2.0, c["y"] + s["y"] / 2.0, c["z"] + s["z"] / 2.0,
        )
        cajas.append({"nombre": b["nombre"], "caja": caja, "dseg": float(dseg(b["tipo"], b.get("valor", 0)))})
    return cajas


def _dist_puntos_caja(puntos: np.ndarray, caja) -> np.ndarray:
    """Distancia de cada punto (n,3) a la superficie de la caja (0 si esta dentro)."""
    lo = np.array(caja[:3])
    hi = np.array(caja[3:])
    d = np.maximum(np.maximum(lo - puntos, 0.0), puntos - hi)
    return np.linalg.norm(d, axis=1)


def _dist_min_segmento_caja(a, b, caja, iteraciones: int = 60) -> float:
    """Distancia minima entre el segmento a-b y la caja. f(t)=dist(a+t(b-a), caja) es convexa,
    asi que la busqueda ternaria da el minimo exacto hasta tolerancia numerica."""
    a = np.asarray(a, float)
    d = np.asarray(b, float) - a

    def f(t):
        return float(_dist_puntos_caja((a + t * d)[None, :], caja)[0])

    lo, hi = 0.0, 1.0
    for _ in range(iteraciones):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    return min(f(0.0), f(1.0), f((lo + hi) / 2.0))


def _holgura_polilinea(puntos, caja) -> float:
    return min(_dist_min_segmento_caja(puntos[i], puntos[i + 1], caja) for i in range(len(puntos) - 1))


# Simplificar la ruta (string pulling)


def simplificar_ruta(ruta, cajas, radio_mm: float, margen_extra_mm: float = 0.0) -> np.ndarray:
    """Quita vertices intermedios mientras la recta directa conserve la holgura que ya tenia la ruta.

    Para cada bloque la holgura exigida es min(radio + zona_de_seguridad, holgura_que_ya_tenia_la_ruta)
    (+ margen_extra). El atajo nunca se acerca mas que la ruta original ni invade la zona de
    seguridad que la ruta si respetaba. Siempre es valido conservar el segmento original."""
    pts = _como_puntos(ruta)
    if len(pts) <= 2 or not cajas:
        return pts if cajas else pts[[0, -1]]  # sin obstaculos es una linea recta

    exigida = []
    for c in cajas:
        previa = _holgura_polilinea(pts, c["caja"])
        exigida.append(min(radio_mm + c["dseg"], previa) + margen_extra_mm)

    def libre(a, b):
        return all(_dist_min_segmento_caja(a, b, c["caja"]) >= e - 1e-6 for c, e in zip(cajas, exigida))

    salida = [pts[0]]
    i = 0
    n = len(pts)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not libre(pts[i], pts[j]):
            j -= 1
        salida.append(pts[j])
        i = j
    return np.array(salida)


# Redondear esquinas con arcos de radio exacto

def _unitario(v):
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


def redondear_esquinas(ruta, radio_mm: float, paso_mm: float = 5.0):
    """Polilinea: (curva (n,3), esquinas). En cada vertice se mete un arco tangente de radio
    radio_mm si cabe, en el caso contrario, el mayor que quepa y la esquina queda marcada factible=False.

    esquinas: [{"vertice", "angulo_giro_deg", "radio_pedido_mm", "radio_logrado_mm", "factible"}]
    Los puntos de la curva estan sobre rectas y circulos, asi que el radio de
    curvatura medido sobre ellos es el de diseño."""
    pts = _como_puntos(ruta)
    n = len(pts)
    largos = np.linalg.norm(np.diff(pts, axis=0), axis=1)

    esquinas, arcos, en_u = [], {}, set()
    for i in range(1, n - 1):
        u = _unitario(pts[i] - pts[i - 1])
        v = _unitario(pts[i + 1] - pts[i])
        theta = math.acos(max(-1.0, min(1.0, float(np.dot(u, v)))))
        if theta < 1e-6:
            continue  # recto o sea que no hay esquina
        # tramos extremos son solo de este vertice. Los intermedios se reparten a la mitad
        disp_ant = largos[i - 1] * (1.0 if i - 1 == 0 else 0.5)
        disp_sig = largos[i] * (1.0 if i == n - 2 else 0.5)
        t_disp = min(disp_ant, disp_sig)
        if theta > math.pi - 1e-3:  # vuelta en U: tan(theta/2) tiende infinito
            t, r_log = 0.0, 0.0
            en_u.add(i)
        else:
            t_nec = radio_mm * math.tan(theta / 2.0)
            t = min(t_nec, t_disp)
            r_log = t / math.tan(theta / 2.0)
        esquinas.append({
            "vertice": i,
            "angulo_giro_deg": round(math.degrees(theta), 1),
            "radio_pedido_mm": float(radio_mm),
            "radio_logrado_mm": round(float(r_log), 2),
            "factible": bool(r_log >= radio_mm - 1e-9),
        })
        if t > _EPS:
            arcos[i] = (u, v, theta, t, r_log)

    def recta(a, b):
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / paso_mm)))
        return [a + (b - a) * s / k for s in range(1, k + 1)]

    curva = [pts[0]]
    actual = pts[0]
    for i in range(1, n - 1):
        if i not in arcos:
            if i in en_u:  # vuelta en U sin arco
                curva += recta(actual, pts[i])
                actual = pts[i]
            continue
        u, v, theta, t, r = arcos[i]
        a_ = pts[i] - u * t
        curva += recta(actual, a_)
        normal = _unitario(v - np.dot(v, u) * u)  # hacia el lado al que gira
        centro = a_ + normal * r
        k = max(2, int(math.ceil(r * theta / paso_mm)))
        for s in range(1, k + 1):
            fi = theta * s / k
            curva.append(centro + r * (-normal * math.cos(fi) + u * math.sin(fi)))
        actual = curva[-1]
    curva += recta(actual, pts[-1])
    curva[-1] = pts[-1].copy()
    return np.array(curva), esquinas


# Revision contra el entorno

def _dist_min_segmentos_caja(puntos: np.ndarray, caja, iteraciones: int = 50) -> float:
    """Distancia minima de toda la polilinea (todos los segmentos) a la caja, con la misma busqueda
    ternaria de _dist_min_segmento_caja pero vectorizada sobre los segmentos."""
    a = puntos[:-1]
    d = puntos[1:] - a
    lo = np.zeros(len(a))
    hi = np.ones(len(a))

    def f(t):
        return _dist_puntos_caja(a + t[:, None] * d, caja)

    for _ in range(iteraciones):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        menor = f(m1) < f(m2)
        hi = np.where(menor, m2, hi)
        lo = np.where(menor, lo, m1)
    return float(min(f(lo).min(), f(hi).min(), f((lo + hi) / 2.0).min()))


def holguras_mm(curva: np.ndarray, cajas, radio_mm: float) -> dict:
    """{nombre: holgura} = distancia minima curva->bloque menos el radio del ducto.
    Negativa = el ducto choca con el bloque. Se mide sobre los segmentos, no solo los vertices.
    Si varias cajas comparten nombre, la holgura del nombre es la minima de todas sus cajas."""
    h: dict = {}
    for c in cajas:
        d = round(_dist_min_segmentos_caja(curva, c["caja"]) - radio_mm, 2)
        h[c["nombre"]] = min(h.get(c["nombre"], d), d)
    return h


def revisar_entorno(curva, cajas, radio_mm: float) -> dict:
    h = holguras_mm(curva, cajas, radio_mm)
    dseg: dict = {}
    for c in cajas:
        dseg[c["nombre"]] = max(dseg.get(c["nombre"], 0.0), c["dseg"])
    colisiones = [n for n, v in h.items() if v < -1e-6]
    fuera_zona = [n for n, v in h.items() if dseg[n] > 0 and -1e-6 <= v < dseg[n] - 1e-6]
    return {"holguras_mm": h, "colisiones": colisiones, "dentro_de_zona_de_seguridad": fuera_zona}



# Tramos rectos en los puertos

def _con_extremos(simple, prefijo, sufijo) -> np.ndarray:
    simple = np.asarray(simple, dtype=float)
    partes = []
    if prefijo is not None and len(prefijo):
        partes.append(np.asarray(prefijo, dtype=float).reshape(-1, 3))
    partes.append(simple)
    if sufijo is not None and len(sufijo):
        partes.append(np.asarray(sufijo, dtype=float).reshape(-1, 3))
    return _como_puntos(np.vstack(partes)) if len(partes) > 1 else simple


def tramos_de_puertos(inicio, fin, puertos) -> dict:
    """Puntos donde termina el tramo recto de cada puerto, y las longitudes.

    puertos = {"inicio": {"normal": [x,y,z], "tramo_mm": L}, "fin": {...}}: `normal` es la normal hacia afuera del
    ducto en ese puerto. El ducto sale del puerto en la direccion contraria durante tramo_mm.
    Cualquiera de los dos puertos puede faltar o tener tramo_mm = 0 (sin tramo recto ahi).
    
    Opcional por puerto: "seccion_constante_mm" (<= tramo_mm; por defecto = tramo_mm) = mm donde la seccion no cambia.
    
    Devuelve {"inicio_astar", "fin_astar", "rectos_mm": (l0, l1)}"""
    inicio, fin = np.asarray(inicio, dtype=float), np.asarray(fin, dtype=float)
    puertos = puertos or {}
    extremos, largos, constantes = [], [], []
    for nombre, punto in (("inicio", inicio), ("fin", fin)):
        p = puertos.get(nombre) or {}
        largo = float(p.get("tramo_mm") or 0.0)
        if not math.isfinite(largo) or largo < 0:
            raise ValueError(f"tramo_mm del puerto '{nombre}' debe ser un numero >= 0.")
        if largo > 0:
            normal = np.asarray(p.get("normal"), dtype=float)
            if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-6:
                raise ValueError(f"El puerto '{nombre}' tiene tramo recto pero su normal no es un vector 3D valido.")
            punto = punto - normal / np.linalg.norm(normal) * largo   # hacia adentro del ducto
        const = p.get("seccion_constante_mm")
        const = largo if const is None else float(const)
        if not math.isfinite(const) or const < 0:
            raise ValueError(f"seccion_constante_mm del puerto '{nombre}' debe ser un numero >= 0.")
        extremos.append(punto)
        largos.append(largo)
        constantes.append(min(const, largo))
    if np.linalg.norm(extremos[0] - extremos[1]) < 1e-6:
        raise ValueError("Los tramos rectos hacen coincidir el inicio y el fin del recorrido libre.")
    return {"inicio_astar": extremos[0].tolist(), "fin_astar": extremos[1].tolist(), "rectos_mm": (constantes[0], constantes[1])}




# Todo junto



def construir_curva(ruta, bloques, r_min_mm: float, radio_envolvente_mm: float, paso_mm: float = 5.0,
                    margen_radio: float = 1.05, distancia_seguridad=None, prefijo=None, sufijo=None,
                    simplificar: bool = True):
    """ruta del A* + bloques -> (curva, info). 
    prefijo/sufijo: puntos que se agregan despues de simplificar, antes del
    primer y despues del ultimo punto de la ruta o sea los tramos rectos de los puertos.

    simplificar=False: la ruta se respeta tal cual. 
    Sin bloques, simplificar_ruta la reduce a una recta entre sus extremos si no hay obstaculos entonces la linea es recta. 
    El radio de diseño es r_min*margen_radio. Si la curva choca o alguna esquina no cabe, reintenta con mas margen
    en los atajos; si nada funciona devuelve el mejor intento, con info['factible'] = False."""
    cajas = cajas_desde_bloques(bloques, distancia_seguridad)
    r_diseno = float(r_min_mm) * margen_radio

    mejor = None
    for factor in ((0.0, 0.25, 0.5, 1.0) if simplificar else (0.0,)):
        simple = (simplificar_ruta(ruta, cajas, radio_envolvente_mm, margen_extra_mm=factor * r_diseno) if simplificar
                  else _como_puntos(ruta))
        simple = _con_extremos(simple, prefijo, sufijo)
        curva, esquinas = redondear_esquinas(simple, r_diseno, paso_mm)
        entorno = revisar_entorno(curva, cajas, radio_envolvente_mm)
        no_caben = [e for e in esquinas if not e["factible"]]
        info = {
            "factible": not no_caben and not entorno["colisiones"],
            "radio_diseno_mm": round(r_diseno, 2),
            "vertices_ruta_original": len(_como_puntos(ruta)),
            "vertices_simplificada": len(simple),
            "margen_extra_mm": round(factor * r_diseno, 2),
            "ruta_simplificada": simple.round(3).tolist(),
            "esquinas": esquinas,
            "esquinas_no_factibles": len(no_caben),
            **entorno,
        }
        puntaje = (len(entorno["colisiones"]), len(no_caben))
        if mejor is None or puntaje < mejor[0]:
            mejor = (puntaje, curva, info)
        if info["factible"]:
            break
    return mejor[1], mejor[2]


def radio_maximo_factible(ruta, bloques, r_max_mm: float, radio_envolvente_mm: float, paso_mm: float = 5.0,
                          distancia_seguridad=None, iteraciones: int = 10, prefijo=None, sufijo=None,
                          simplificar: bool = True):
    """Mayor r_min (en mm, entero hacia abajo) que si cabe en esta ruta y estos bloques, buscado por
    biseccion entre 0 y r_max_mm. Devuelve None si ni un radio chico cabe.
    """
    def cabe(r):
        return construir_curva(ruta, bloques, r, radio_envolvente_mm, paso_mm,
                               distancia_seguridad=distancia_seguridad, prefijo=prefijo, sufijo=sufijo,
                               simplificar=simplificar)[1]["factible"]

    bajo, alto = 1.0, float(r_max_mm)
    if not cabe(bajo):
        return None
    for _ in range(iteraciones):
        medio = (bajo + alto) / 2.0
        if cabe(medio):
            bajo = medio
        else:
            alto = medio
    return int(math.floor(bajo))


def proponer_alternativas(info: dict, r_min_mm: float, r_max_factible_mm) -> dict:
    """Convierte un diagnostico de infactibilidad en propuestas para el usuario. La maquina no
    decide si cambiar r_min o mover bloques ya que son decisiones de diseño para el ususario."""
    props = []
    if r_max_factible_mm:
        props.append({
            "id": "bajar_r_min",
            "descripcion": f"Con este entorno el mayor R_min que cabe es ~{r_max_factible_mm} mm "
                           f"(se exige {r_min_mm:g} mm). Requiere que el responsable apruebe un R_min menor.",
            "requiere_aprobacion": True,
            "parametros": {"r_min_mm": r_max_factible_mm},
        })
    props.append({
        "id": "mover_entorno",
        "descripcion": "Mover/reducir los bloques o los puntos de inicio y fin para abrir mas espacio de giro "
                       "(el A* actual no conoce la curvatura: volver a correrlo con los mismos datos da lo mismo).",
        "requiere_aprobacion": False,
        "parametros": {},
    })
    if not info["colisiones"]:
        props.append({
            "id": "generar_igual",
            "descripcion": "Generar el ducto de todos modos con el mayor radio posible en cada esquina, para "
                           "inspeccionarlo (NO cumple R_min).",
            "requiere_aprobacion": False,
            "parametros": {"generar_si_infactible": True},
        })
    return {"radio_maximo_factible_mm": r_max_factible_mm, "propuestas": props}


def _importar_ruteo():
    for nombre in ("ruteo_astar", "herramientas.ruteo_astar"):
        try:
            return importlib.import_module(nombre)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError("No se encontro ruteo_astar (ni como 'ruteo_astar' ni como 'herramientas.ruteo_astar').")


def ducto_desde_entorno(inicio, fin, ancho_mm: float | None, alto_mm: float | None, espesor_mm: float,
                        parametros: dict | None = None, nombre: str = "ducto",
                        opciones_astar: dict | None = None, exportar_archivos: bool = True,
                        generar_si_infactible: bool = False, campo_costo=None,
                        ancho_fin_mm: float | None = None, alto_fin_mm: float | None = None,
                        ley_transicion: str = "suave", secciones=None, puertos=None, ruta_dada=None) -> dict:
    """Entorno de bloques -> A* -> curva con R_min exacto -> ducto CAD. Un solo reporte (dict).
    No lanza por problemas de ruta: devuelve estado 'sin_ruta' o 'error' con el mensaje del A*.
    campo_costo: opcional, funcion lista-de-puntos -> lista-de-costos en [0,1] que el A* suma a su costo (C_NN).
    ancho_fin_mm/alto_fin_mm: opcionales, seccion del puerto final si es distinta de la inicial
    Para esquivar los bloques se usa la seccion mayor de las dos.
    secciones: [(pos, ancho_mm, alto_mm), ...] perfil por estaciones. Para esquivar los bloques se usa la mayor de todas las secciones.

    puertos: {"inicio": {"normal": [x,y,z], "tramo_mm": L}, "fin": {...}} = tramos rectos en las bocas.

    ruta_dada: lista de puntos [x,y,z] del extremo del tramo inicial al del final 
    Si se da, no se corre el A* y se le redondean los codos con R_min y se revisa contra los bloques, como a cualquier ruta del A*.

    Si la curva con R_min no es factible  devuelve estado 'infactible' sin generar CAD. 
    Con diagnostico, 'propuestas' y una 'pregunta' para el humano el agente anuncia el
    problema y pregunta al usuario que decide hacer 
    Solo se genera el ducto igual si se llama con generar_si_infactible=True.
    """
    parametros = parametros or {}
    ruteo = _importar_ruteo()
    if secciones is not None:
        if any(v is not None for v in (ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm)):
            raise ValueError("Da las secciones por estaciones (secciones=...) o ancho/alto (+ fin), no las dos cosas.")
        _, anchos_c, altos_c = validar_secciones_control(secciones)
        radio_env = float(np.max(np.hypot(anchos_c, altos_c))) / 2.0
        menor = float(min(anchos_c.min(), altos_c.min()))
    else:
        if (ancho_fin_mm is None) != (alto_fin_mm is None):
            raise ValueError("Para cambiar la seccion en el puerto final hay que dar ancho_fin_mm Y alto_fin_mm.")
        a1 = ancho_mm if ancho_fin_mm is None else ancho_fin_mm
        h1 = alto_mm if alto_fin_mm is None else alto_fin_mm
        # tubo que envuelve al rectangulo
        radio_env = max(math.hypot(ancho_mm, alto_mm), math.hypot(a1, h1)) / 2.0
        menor = min(ancho_mm, alto_mm, a1, h1)

    tramos = tramos_de_puertos(inicio, fin, puertos)
    ini_astar, fin_astar = tramos["inicio_astar"], tramos["fin_astar"]
    rectos = tramos["rectos_mm"] if any(tramos["rectos_mm"]) else None
    largo_ini = float(((puertos or {}).get("inicio") or {}).get("tramo_mm") or 0.0)
    largo_fin = float(((puertos or {}).get("fin") or {}).get("tramo_mm") or 0.0)
    prefijo = [list(map(float, inicio))] if largo_ini > 0 else None
    sufijo = [list(map(float, fin))] if largo_fin > 0 else None

    cfg = dict(opciones_astar or {})
    cfg["radio_ducto_mm"] = max(float(cfg.get("radio_ducto_mm", 0.0)), radio_env)
    # campo_costo solo se pasa si hay uno
    if ruta_dada is not None:
        ruta_dada = [list(map(float, p)) for p in ruta_dada]
        if len(ruta_dada) < 2:
            raise ValueError("ruta_dada necesita al menos dos puntos.")
        res = {"estado": "ok", "ruta": ruta_dada, "bloques_cercanos": [], "mensaje": "",
               "longitud_mm": float(sum(np.linalg.norm(np.diff(np.asarray(ruta_dada), axis=0), axis=1)))}
    else:
        res = ruteo.calculo_ruta(ini_astar, fin_astar, cfg, campo_costo=campo_costo) if campo_costo is not None \
            else ruteo.calculo_ruta(ini_astar, fin_astar, cfg)
    if res["estado"] != "ok":
        return {"estado": res["estado"], "mensaje": res["mensaje"]}

    bloques = ruteo.listar_bloques()
    r_min = parametros.get("r_min_mm")
    curva, info = None, None
    if r_min is not None:
        paso = max(1.0, menor / 4.0)
        curva, info = construir_curva(res["ruta"], bloques, float(r_min), radio_env, paso,
                                      distancia_seguridad=getattr(ruteo, "distancia_seguridad_mm", None),
                                      prefijo=prefijo, sufijo=sufijo, simplificar=ruta_dada is None)

    if info is not None and not info["factible"] and not generar_si_infactible:
        r_max = radio_maximo_factible(res["ruta"], bloques, float(r_min), radio_env, paso,
                                      distancia_seguridad=getattr(ruteo, "distancia_seguridad_mm", None),
                                      prefijo=prefijo, sufijo=sufijo, simplificar=ruta_dada is None)
        malas = [e for e in info["esquinas"] if not e["factible"]]
        return {
            "estado": "infactible",
            "mensaje": (f"No se puede cumplir R_min = {float(r_min):g} mm en esta ruta: "
                        f"{len(malas)} esquina(s) no caben y {len(info['colisiones'])} bloque(s) chocan."),
            "diagnostico": {"esquinas_no_factibles": malas, "colisiones": info["colisiones"],
                            "holguras_mm": info["holguras_mm"]},
            **proponer_alternativas(info, float(r_min), r_max),
            "pregunta": "Como quieres proceder? Elige una propuesta (bajar_r_min, mover_entorno o generar_igual).",
        }

    ruta_final = _con_extremos(res["ruta"], prefijo, sufijo) if curva is None else res["ruta"]  # sin R_min se utiliza suavizado Chaikin
    reporte = generar_ducto(ruta_final, ancho_mm, alto_mm, espesor_mm, parametros=parametros,
                            nombre=nombre, exportar_archivos=exportar_archivos, curva=curva,
                            ancho_fin_mm=ancho_fin_mm, alto_fin_mm=alto_fin_mm, ley_transicion=ley_transicion,
                            secciones=secciones, rectos_mm=rectos)
    reporte["estado"] = "ok"
    if prefijo or sufijo:
        reporte["tramos_rectos_mm"] = [largo_ini, largo_fin]
    reporte["ruta_astar"] = {"puntos": len(res["ruta"]), "longitud_mm": res["longitud_mm"],
                             "bloques_cercanos": res["bloques_cercanos"]}
    if info is not None:
        reporte["curva"] = {k: v for k, v in info.items() if k != "esquinas"}
        reporte["curva"]["esquinas_resumen"] = info["esquinas"]
    else:
        reporte["curva"] = {"aviso": "No hay r_min aprobado: se uso suavizado Chaikin, sin revisar colisiones."}
    return reporte
