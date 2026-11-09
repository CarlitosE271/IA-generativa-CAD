"""
Evalua el pipeline contra los ductos de referencia, sin entorno 
Se recomienda ver src/herramientas/evaluacion_referencias.py.

Uso:
    python scripts/evaluar_referencias.py                       # las 3 piezas, con CadQuery
    python scripts/evaluar_referencias.py --sin-cad             # malla con numpy, sin CadQuery 
    python scripts/evaluar_referencias.py segunda_pieza --ley lineal
    python scripts/evaluar_referencias.py --solo-puertos        # Solo las secciones de los 2 puertos
    python scripts/evaluar_referencias.py --ruta-referencia     # el generador sigue la linea central de la referencia
    python scripts/evaluar_referencias.py --sin-tramos          # sin tramos rectos en los puertos (modo anterior)
    python scripts/evaluar_referencias.py --seccion primera_pieza:2=150x60     # corregir un puerto con la medida del CAD
    python scripts/evaluar_referencias.py --r-min 60 --area-min 3000 --area-max 8000 --aspecto-max 4

Los parametros salen de data/parametros_aprobados.json salvo los que se den por linea de comandos. 

Por defecto el ducto generado lleva ademas 10 secciones aproximadamente del cuerpo copiadas de la referencia (perfil por estaciones,
interpoladas con PCHIP).

Escribe en data/evaluacion/<pieza>/ : evaluacion.json, ficha.png y el STL generado.
"""
import argparse
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from cad import ducto  # noqa: E402
from herramientas import evaluacion_referencias as ev  # noqa: E402
from herramientas import ficha_ducto as fd  # noqa: E402
from herramientas import ingesta_malla as im  # noqa: E402


def _parametros(args) -> dict:
    ruta = RAIZ / "data" / "parametros_aprobados.json"
    p = {}
    if ruta.exists():
        try:
            p = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"AVISO: no se pudo leer {ruta}.")
    for clave, valor in (("r_min_mm", args.r_min), ("area_min_mm2", args.area_min), ("area_max_mm2", args.area_max),
                         ("aspect_ratio_max", args.aspecto_max)):
        if valor is not None:
            p[clave] = valor
    return p


def _secciones(lista) -> dict:
    """['primera_pieza:2=150x60'] -> {'primera_pieza': {2: (150.0, 60.0)}}"""
    salida: dict = {}
    for texto in lista or []:
        m = re.fullmatch(r"([^:]+):([12])=(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", texto.strip())
        if not m:
            raise SystemExit(f"--seccion mal escrito: {texto!r}. Formato: NOMBRE:PUERTO=ANCHOxALTO, p. ej. primera_pieza:2=150x60")
        salida.setdefault(m.group(1), {})[int(m.group(2))] = (float(m.group(3)), float(m.group(4)))
    return salida


def _dimensiones(esc) -> dict:
    """Argumentos de seccion para el pipeline: por estaciones si el escenario las trae, si no las dos de los puertos."""
    if esc.get("secciones"):
        return {"ancho_mm": None, "alto_mm": None, "ancho_fin_mm": None, "alto_fin_mm": None, "secciones": esc["secciones"]}
    return {"ancho_mm": esc["ancho_mm"], "alto_mm": esc["alto_mm"], "ancho_fin_mm": esc["ancho_fin_mm"],
            "alto_fin_mm": esc["alto_fin_mm"], "secciones": None}


def _exigir_curva_valida(curva, params, nombre):
    """La curva que se le da al generador debe cumplir R_min: si no, el loft liso ondula en los quiebres y la
    evaluacion mide un artefacto, no el CAD. (La malla numpy es lineal y absorbe quiebres: sin esta guarda la prueba en
    seco no los ve.)"""
    import numpy as np

    r_min = params.get("r_min_mm")
    if r_min is None:
        return
    r = ducto.validar_radio_minimo(np.asarray(curva, dtype=float), float(r_min))
    if not r["cumple"]:
        raise RuntimeError(f"{nombre}: la curva dada al generador no cumple R_min = {r_min} mm (radio minimo "
                           f"{r['radio_minimo_mm']} mm a {r['posicion_peor_mm']} mm del inicio, {r['num_violaciones']} puntos). "
                           f"Un quiebre aqui hace ondular el loft: no se evalua.")


def _curva_sin_entorno(esc, params):
    """(curva, rectos_mm) del ducto SIN obstaculos y sin CadQuery, por el MISMO camino que ducto_desde_entorno:
    tramos rectos -> ruta (recta del A* sin bloques, o la ruta dada) -> codos con arcos de r_min x 1.05."""
    from cad import curva_ruta

    pd = esc.get("puertos_ducto") or {}
    t = curva_ruta.tramos_de_puertos(esc["inicio"], esc["fin"], pd)
    rectos = t["rectos_mm"] if any(t["rectos_mm"]) else None
    if esc.get("curva"):                       # la linea central de la referencia, ya alisada: se usa tal cual
        import numpy as np
        _exigir_curva_valida(esc["curva"], params, esc["nombre"])
        return np.array(esc["curva"]), rectos
    ruta = [t["inicio_astar"], t["fin_astar"]]
    prefijo = [esc["inicio"]] if (pd.get("inicio") or {}).get("tramo_mm") else None
    sufijo = [esc["fin"]] if (pd.get("fin") or {}).get("tramo_mm") else None
    r_min = params.get("r_min_mm")
    if r_min is None:
        return ducto.suavizar_ruta(curva_ruta._con_extremos(ruta, prefijo, sufijo), paso_mm=5.0), rectos
    curva, _ = curva_ruta.construir_curva(ruta, [], float(r_min), 1.0, 5.0, prefijo=prefijo, sufijo=sufijo,
                                          simplificar=True)
    return curva, rectos


def _generar_con_cad(ley: str):
    from cad import curva_ruta  # noqa: E402

    def generar(esc, params):
        if esc.get("curva"):   # ruta de la referencia: no hay A* ni esquinas que redondear, la curva ya esta hecha
            import numpy as np
            dim = _dimensiones(esc)
            _exigir_curva_valida(esc["curva"], params, esc["nombre"])
            rep = ducto.generar_ducto(None, dim.pop("ancho_mm"), dim.pop("alto_mm"), esc["espesor_mm"], parametros=params,
                                      nombre="eval_" + re.sub(r"[^A-Za-z0-9_-]", "_", esc["nombre"])[:50],
                                      curva=np.array(esc["curva"]), ley_transicion=ley,
                                      rectos_mm=tuple(esc["rectos_mm"]) if esc.get("rectos_mm") else None, **dim)
            if not rep.get("solido_valido", True):
                raise RuntimeError(f"{esc['nombre']}: CadQuery devolvio un solido NO valido (isValid = False).")
            return ducto.ruta_cad / rep["archivos"]["stl"]
        ruteo = curva_ruta._importar_ruteo()
        anterior = getattr(ruteo, "RUTA_ENTORNO", None)
        with tempfile.TemporaryDirectory() as tmp:
            ruteo.configurar_ruta_entorno(str(Path(tmp) / "entorno.json"))  # entorno VACIO
            try:
                dim = _dimensiones(esc)
                rep = curva_ruta.ducto_desde_entorno(
                    esc["inicio"], esc["fin"], dim.pop("ancho_mm"), dim.pop("alto_mm"), esc["espesor_mm"], parametros=params,
                    nombre="eval_" + re.sub(r"[^A-Za-z0-9_-]", "_", esc["nombre"])[:50],
                    ley_transicion=ley, puertos=esc.get("puertos_ducto"), **dim)
            finally:
                if anterior:
                    ruteo.configurar_ruta_entorno(anterior)
        if rep.get("estado") != "ok":
            raise RuntimeError(f"El pipeline no genero el ducto: {rep.get('estado')} - {rep.get('mensaje')}")
        return ducto.ruta_cad / rep["archivos"]["stl"]

    return generar


def _generar_sin_cad(ley: str, carpeta: Path):
    def generar(esc, params):
        # sin bloques el A* devuelve la recta (ruteo_astar: "sin bloques es linea recta"): se usa igual
        curva, rectos = _curva_sin_entorno(esc, params)
        dim = _dimensiones(esc)
        tri = ducto.malla_numpy(curva, dim.pop("ancho_mm"), dim.pop("alto_mm"), esc["espesor_mm"], max_estaciones=40,  # = valor por defecto de construir_ducto (CadQuery)
                                ley_transicion=ley, rectos_mm=rectos, **dim)
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / "ducto_numpy.stl"
        im.escribir_stl(ruta, tri)
        return ruta

    return generar


def _figura_comparacion(ref: dict, gen: dict, destino: Path, titulo: str, nota: str = "") -> None:
    """Referencia (gris/azul) contra el ducto generado (rojo): lineas centrales y perfiles de seccion, para el panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def datos(f, invertir=False):
        q = f["perfil"]
        L = f["longitud_linea_central_mm"]
        u = np.array([p["s_mm"] / L for p in q])
        c = np.array([p["centro_mm"] for p in q])
        a = np.array([p["ancho_int_mm"] for p in q])
        h = np.array([p["alto_int_mm"] for p in q])
        ar = np.array([p["area_int_mm2"] for p in q])
        if invertir:
            u, c, a, h, ar = 1 - u[::-1], c[::-1], a[::-1], h[::-1], ar[::-1]
        return u, c, a, h, ar

    # el sentido de la linea central de la ficha es arbitrario: se alinea con el de la referencia por cercania
    pr, pg = ref["perfil"], gen["perfil"]
    d = lambda x, y: float(np.linalg.norm(np.array(x) - np.array(y)))  # noqa: E731
    invertir = d(pg[0]["centro_mm"], pr[0]["centro_mm"]) > d(pg[0]["centro_mm"], pr[-1]["centro_mm"])
    ur, cr, ar_, hr, areaR = datos(ref)
    ug, cg, ag, hg, areaG = datos(gen, invertir)
    fig, ejes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (i, j, nombre) in zip(ejes[0], [(0, 1, "Planta (X-Y)"), (0, 2, "Perfil (X-Z)")]):
        ax.plot(cr[:, i], cr[:, j], "-", c="tab:blue", lw=2.2, label="referencia")
        ax.plot(cg[:, i], cg[:, j], "-", c="tab:red", lw=1.6, label="generado")
        ax.scatter([cr[0, i], cr[-1, i]], [cr[0, j], cr[-1, j]], c="tab:blue", s=40, zorder=5)
        ax.set_aspect("equal")
        ax.set_title(nombre + " - linea central")
        ax.set_xlabel("mm")
        ax.legend(fontsize=8)
    ax = ejes[1][0]
    ax.plot(ur, areaR, c="tab:blue", lw=1.6, label="referencia")
    ax.fill_between(ur, areaR * 0.9, areaR * 1.1, color="tab:blue", alpha=0.15, label="referencia +-10 % (incertidumbre de la ficha)")
    ax.plot(ug, areaG, c="tab:red", lw=1.6, label="generado")
    ax.set_title("Area de paso (interior) a lo largo del ducto")
    ax.set_xlabel("distancia relativa u = s / largo")
    ax.set_ylabel("mm2")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax = ejes[1][1]
    ax.plot(ur, ar_, c="tab:blue", lw=1.6, label="ancho ref")
    ax.plot(ur, hr, c="tab:blue", lw=1.6, ls="--", label="alto ref")
    ax.plot(ug, ag, c="tab:red", lw=1.4, label="ancho gen")
    ax.plot(ug, hg, c="tab:red", lw=1.4, ls="--", label="alto gen")
    ax.set_title("Ancho y alto interiores")
    ax.set_xlabel("distancia relativa u")
    ax.set_ylabel("mm")
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle(titulo + (("\n" + nota) if nota else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def _imprimir(res: dict) -> None:
    print(f"\n=== {res['escenario']} ===")
    for a in res["advertencias"]:
        print(f"  ATENCION: {a}")
    print("  Cumplimiento (lo medido en el ducto generado):")
    for k, c in res["cumplimiento"].items():
        if k == "nota":
            continue
        medido = next((v for kk, v in c.items() if kk.startswith("medido")), None)
        limite = next((v for kk, v in c.items() if kk.startswith("limite")), None)
        print(f"    {k:<18} medido {medido}  limite {limite}  ->  {c['cumple']}")
    print("  Parecido con la referencia en lo comparable sin entorno (razon generada/referencia):")
    for k, v in res["comparable"].items():
        print(f"    {k:<30} ref {v['referencia']!s:>8}  gen {v['generada']!s:>8}  razon {v['razon']}")
    if res.get("ruta_de_referencia"):
        print("  Con la RUTA de la referencia dada, largo y radios SI se pueden comparar (misma ruta; mide la fidelidad del CAD, no el ruteo):")
    else:
        print("  NO comparable sin entorno (dependen de los obstaculos; la recta no es 'peor' ni 'mejor'):")
    for k, v in res["no_comparable_sin_entorno"].items():
        print(f"    {k:<30} ref {v['referencia']!s:>8}  gen {v['generada']!s:>8}")
    pf = res.get("perfil")
    if pf:
        print(f"  Perfil de area a lo largo del ducto (modo: {res['modo_seccion']}; u de {pf['u_desde_hasta'][0]} a {pf['u_desde_hasta'][1]}):")
        print(f"    error de area   mediana {pf['error_area_mediana']:.1%}   p90 {pf['error_area_p90']:.1%}   "
              f"(la ficha mide el area con ~10 % de incertidumbre)")
        print(f"    error de aspecto mediana {pf['error_aspecto_mediana']:.1%}   p90 {pf['error_aspecto_p90']:.1%}")
        print(f"    area minima generada / area minima de la referencia: {pf['area_min_generada_sobre_referencia']}")
    r = res.get("tramos_rectos_mm")
    print("  Ajuste a los puertos (" + (f"tramos rectos de {r[0]:g} y {r[1]:g} mm que imponen la direccion de salida" if r
                                       else "SIN tramos rectos: la normal del puerto NO se impone") + "):")
    for p in res["puertos"]:
        print(f"    {p['puerto_referencia']}: a {p['distancia_mm']} mm del puerto de la referencia "
              f"(lateral {p['lateral_mm']}, axial {p['axial_mm']}: el axial tiene ~15 mm de holgura de medicion), "
              f"normal desviada {p['angulo_normal_grados']} grados")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("piezas", nargs="*", help="nombres de carpeta en data/referencia (por defecto, todas las que tengan ficha.json)")
    ap.add_argument("--referencias", default=str(RAIZ / "data" / "referencia"))
    ap.add_argument("--salida", default=str(RAIZ / "data" / "evaluacion"))
    ap.add_argument("--ley", choices=ducto.LEYES_TRANSICION, default="suave", help="ley de transicion de seccion")
    ap.add_argument("--sin-cad", action="store_true", help="usa la malla numpy en vez de CadQuery (prueba en seco)")
    ap.add_argument("--solo-puertos", action="store_true",
                    help="no copia el perfil del cuerpo de la referencia: solo las secciones de los dos puertos (modo anterior)")
    ap.add_argument("--sin-tramos", action="store_true",
                    help="no pone tramos rectos en los puertos (el ducto puede salir de la boca en cualquier direccion)")
    ap.add_argument("--factor-tramo", type=float, default=ev.FACTOR_TRAMO,
                    help="largo del tramo recto en diametros hidraulicos del puerto (por defecto 1.5; minimo 85 mm)")
    ap.add_argument("--ruta-referencia", action="store_true",
                    help="en vez del A* (recta sin obstaculos) usa la linea central medida de la referencia como ruta: "
                         "prueba que el generador SIGUE una ruta dada. Escribe en data/evaluacion/<pieza>_rutaref/")
    ap.add_argument("--paso", type=float, default=3.0, help="voxel de la ficha en mm")
    ap.add_argument("--seccion", action="append", help="NOMBRE:PUERTO=ANCHOxALTO (interior, mm); se puede repetir")
    ap.add_argument("--r-min", type=float, default=None)
    ap.add_argument("--area-min", type=float, default=None)
    ap.add_argument("--area-max", type=float, default=None)
    ap.add_argument("--aspecto-max", type=float, default=None)
    args = ap.parse_args()

    carpetas = [Path(args.referencias) / n for n in args.piezas] if args.piezas else ev.carpetas_de_referencia(args.referencias)
    if not carpetas:
        print(f"No hay fichas en {args.referencias}. Corre antes:  python scripts/ficha_ducto.py <STL de las referencias>")
        return 1
    params = _parametros(args)
    correcciones = _secciones(args.seccion)
    print(f"Parametros usados: {params if params else 'ninguno (todo saldra no_verificado)'}   "
          f"ley: {args.ley}   {'malla numpy (sin CadQuery)' if args.sin_cad else 'CadQuery'}")

    todos, codigo = {}, 0
    for carpeta in carpetas:
        try:
            ref = json.loads((carpeta / "ficha.json").read_text(encoding="utf-8"))
            esc = ev.escenario_desde_ficha(ref, carpeta.name + ("_rutaref" if args.ruta_referencia else ""),
                                           correcciones.get(carpeta.name),
                                           con_estaciones=not args.solo_puertos,
                                           factor_tramo=None if args.sin_tramos else args.factor_tramo,
                                           ruta_referencia=args.ruta_referencia, r_min_mm=params.get("r_min_mm"))
            destino = Path(args.salida) / (carpeta.name + ("_rutaref" if args.ruta_referencia else ""))
            generar = _generar_sin_cad(args.ley, destino) if args.sin_cad else _generar_con_cad(args.ley)

            def medir(stl):
                return fd.ficha_de_malla(im.leer_stl(stl), paso_mm=args.paso, espesor_pared_mm=esc["espesor_mm"])

            res = ev.evaluar_escenario(esc, ref, params, generar, medir)
        except (OSError, ValueError, KeyError, RuntimeError, im.IngestaError, ModuleNotFoundError) as e:
            print(f"\nERROR en {carpeta.name}: {e}")
            codigo = 1
            continue
        destino.mkdir(parents=True, exist_ok=True)
        salida = {"escenario_pedido": {k: v for k, v in esc.items() if k != "curva"}, "parametros_usados": params, "ley_transicion": args.ley,
                  "con_cad": not args.sin_cad, "solo_puertos": args.solo_puertos, "sin_tramos": args.sin_tramos, "ruta_referencia": args.ruta_referencia, **ev.a_json(res)}
        (destino / "evaluacion.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            spec = importlib.util.spec_from_file_location("ficha_cli", RAIZ / "scripts" / "ficha_ducto.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod._figura(im.leer_stl(res["stl"]), res["_ficha"], destino / "ficha.png", carpeta.name + " (generado)")
            nota = ("Puerto(s) " + ", ".join(map(str, res["puertos_de_referencia_corregidos_con_cad"])) +
                    " de la referencia corregido(s) con la medida del CAD (la ficha los midio con orejas)"
                    if res["puertos_de_referencia_corregidos_con_cad"] else "")
            _figura_comparacion(ref, res["_ficha"], destino / "comparacion.png",
                                f"{carpeta.name}: referencia vs ducto generado ({res['modo_seccion']}); error de area mediana "
                                f"{res['perfil']['error_area_mediana']:.1%}", nota)
        except ImportError:
            print("(matplotlib no esta instalado: se omite la figura)")
        _imprimir(res)
        print(f"  escrito: {destino / 'evaluacion.json'}")
        todos[carpeta.name] = ev.a_json(res)
    if todos:
        Path(args.salida).mkdir(parents=True, exist_ok=True)
        (Path(args.salida) / ("resumen_rutaref.json" if args.ruta_referencia else "resumen.json")).write_text(json.dumps(todos, indent=2, ensure_ascii=False), encoding="utf-8")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
