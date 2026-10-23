#!/usr/bin/env python3
"""Arma la copia pública del curso en ``_publicar/curso/`` (espejo de ``curso/`` en GitHub).

    python curso/scripts/publicar_web.py                 # reconstruye todo (idempotente)
    python curso/scripts/publicar_web.py --solo cuadernos
    python curso/scripts/publicar_web.py --solo diapositivas --solo leeme

El destino es la carpeta ``curso/`` del repositorio público
``jalonso1979/puremacro``. Este script **no** hace commit ni push: sólo deja la
carpeta lista para copiarla allí. Se puede correr cada semana; lo que no cambió
no se reescribe (ni cambia de fecha), y lo que desapareció de la fuente se borra
del destino.

Qué construye:

``notebooks/``
    Todos los ``T*.ipynb`` (inglés y ``*_es``) con sus salidas ejecutadas, los
    auxiliares ``_datos.py``, ``_nbstyle.py``, ``_tutor.py``, ``_llm.py`` y ``_t01.py``,
    ``data_curso/`` completo (archivos reales, no enlaces; sin ``tmp_check/``) y de
    ``modelos/`` sólo lo que los cuadernos leen: los ``.mod`` y los ``*_results.mat`` que
    algún cuaderno nombra. En las copias —nunca en los maestros— se reescriben
    las dos primeras celdas: la insignia de Colab (celda 0) y la celda de
    configuración (celda 1), que en Colab instala ``puremacro`` y hace un clon
    parcial del repositorio público en lugar de montar Google Drive.
    ``T04_B_dashboard_rbc_interactive.py`` no se publica: sólo lanza
    ``dashboards/rbc_dashboard.py``, que vive fuera de ``notebooks/``.
``notebooks/README.md`` y ``notebooks/data_curso/FUENTES.md``
    Cómo abrir cada cuaderno (tabla con enlaces a Colab) y de dónde sale cada
    archivo de datos, agrupado por editor.
``site/diapositivas/``
    Los ocho mazos ``Slides0Xmav.pdf`` con nombres legibles sacados del
    ``\\title`` de cada ``.tex``, más ``index.json``. El resto de ``site/`` lo
    escribe otro proceso: aquí no se toca.

Al final imprime el tamaño total y avisa de cualquier archivo de más de 50 MB
(GitHub advierte a partir de 50 MB y rechaza a partir de 100 MB).
"""
from __future__ import annotations

import argparse
import ast
import csv
import filecmp
import json
import os
import re
import shutil
import shlex
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FUENTE_NB = RAIZ / "notebooks"
# En puremacro escribe en build/curso para mantener limpio el árbol de git;
# si MAV existe y corre en MAV, usa _publicar/curso.
DESTINO = Path(os.environ.get(
    "PUBLICAR_DESTINO",
    RAIZ / "_publicar" / "curso" if (RAIZ / "Slides01mav.pdf").is_file()
    else RAIZ.parent / "build" / "curso"
))
DEST_NB = DESTINO / "notebooks"
DEST_DIAP = DESTINO / "site" / "diapositivas"

REPO = "jalonso1979/puremacro"
EDICION = "course-2026A-5.5.0"
RAMA = EDICION
RUTA_EN_REPO = "curso/notebooks"
# Versión de puremacro que piden los cuadernos en Colab.
PUREMACRO = "puremacro==5.5.0"

# _tutor.py importa _llm.py cuando se llama a tutor(), no al importarse: el paquete lleva los dos o cada
# pregunta al tutor termina en el texto de respaldo con «ModuleNotFoundError».
AUXILIARES = ["_datos.py", "_nbstyle.py", "_tutor.py", "_llm.py", "_t01.py"]
EXCLUIR_DATOS = {"tmp_check", "__pycache__", ".ipynb_checkpoints"}
# Cuadernos que existen en notebooks/ pero aún no se publican.
EXCLUIDOS: set[str] = set()  # T00_B runs in cache mode; live setup is optional.
NO_PUBLICAR_AUN: set[str] = EXCLUIDOS
# Notas internas del paquete de datos (menciona claves de calificación y figuras_curso.py).
EXCLUIR_ARCHIVOS_DATOS = {Path("bundle_2026A/LEEME.md")}

# Limpieza de las copias públicas (decidida por el profesor el 2026-09-16); los maestros no se tocan.
# 1. Las celdas de soluciones de los ejercicios no se publican.
ENCABEZADOS_SOLUCIONES = ("### Soluciones (esquema)", "### Solutions (outline)")
# 2. Referencias a pendientes internos del .tex de las diapositivas, reescritas en voz de lección.
#    Si un texto ya no aparece, se avisa: el cuaderno maestro cambió y hay que revisar la regla.
REESCRITURAS_INTERNAS = {
    "T04_C_contabilidad_ciclo_es.ipynb": [
        ("El pie de la figura del mazo A4 lo declara en voz alta y el `.tex` lo lleva escrito como pendiente "
         "desde hace dos cursos: ", "El pie de la figura del mazo A4 lo declara en voz alta: "),
        ("### 6.3 La respuesta al pendiente del mazo", "### 6.3 ¿Puede México entrar en la figura del mazo?"),
        ("El `TODO` del `.tex` pide sustituir a España por México en `fig_bca_cunas_gr_covid`. ",
         "¿Se puede sustituir a España por México en la figura de cuñas del mazo? "),
        (" Lo que hay que decir en clase es esto.", ""),
    ],
    "T04_C_business_cycle_accounting.ipynb": [
        ("The caption of deck A4's figure declares it out loud and the `.tex` has carried it as a pending item "
         "for two courses: ", "The caption of deck A4's figure declares it out loud: "),
        ("### 6.3 The answer to the deck's pending item", "### 6.3 Can Mexico enter the deck's figure?"),
        ("The `TODO` in the `.tex` asks for Spain to be replaced by Mexico in `fig_bca_cunas_gr_covid`. ",
         "Can Mexico replace Spain in the deck's wedge figure? "),
        (" What has to be said in class is this.", ""),
    ],
}
# 3. Rutas locales en las salidas guardadas: se vuelven relativas a curso/notebooks o al paquete.
_PM_ROOT = str(Path(__file__).resolve().parents[2]) + "/"
RUTAS_LOCALES = [
    (str(Path(__file__).resolve().parent.parent / "notebooks") + "/", ""),
    (_PM_ROOT, ""),
]
MB = 1024 * 1024
AVISO_MB = 50
LIMITE_MB = 100


# ---------------------------------------------------------------------------
# Utilidades de escritura idempotente
# ---------------------------------------------------------------------------
def _escribir_texto(ruta: Path, texto: str) -> bool:
    """Escribe ``texto`` sólo si difiere de lo que ya hay. Devuelve si escribió."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.is_file() and ruta.read_text(encoding="utf-8") == texto:
        return False
    ruta.write_text(texto, encoding="utf-8")
    return True


def _copiar_archivo(src: Path, dst: Path) -> bool:
    """Copia el contenido real (sigue enlaces) sólo si el destino difiere."""
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and filecmp.cmp(src, dst, shallow=False):
        return False
    shutil.copy2(src, dst)
    return True


def _podar(carpeta: Path, conservar: set[Path]) -> list[Path]:
    """Borra de ``carpeta`` todo archivo que no esté en ``conservar`` (y los directorios vacíos)."""
    borrados = []
    if not carpeta.is_dir():
        return borrados
    for p in sorted(carpeta.rglob("*"), reverse=True):
        if p.is_file() or p.is_symlink():
            if p not in conservar:
                p.unlink()
                borrados.append(p)
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    return borrados


def _es_espanol(nombre: str) -> bool:
    return Path(nombre).stem.endswith("_es")


def _codigo(nombre: str) -> str:
    """``T05_A_svar_identification_es.ipynb`` -> ``T05_A``; ``T00_syllabus.ipynb`` -> ``T00``."""
    m = re.match(r"(T\d\d(?:_[A-Z])?)(?:_|\.|$)", nombre)
    return m.group(1) if m else Path(nombre).stem


def _url_colab(nombre: str) -> str:
    return f"https://colab.research.google.com/github/{REPO}/blob/{RAMA}/{RUTA_EN_REPO}/{nombre}"


# ---------------------------------------------------------------------------
# Celda 0: insignia de Colab
# ---------------------------------------------------------------------------
_INSIGNIA = re.compile(
    r"\[!\[[^\]]*\]\(https://colab\.research\.google\.com/assets/colab-badge\.svg\)\]\([^)]*\)")
_NOTA_COLAB = re.compile(r"^> \*\*(?:Nota para Google Colab|Google Colab [Nn]ote)\*\*:.*$", re.M)

NOTA_ES = ("> **Nota para Google Colab**: la celda siguiente instala `puremacro` y, si hace falta, "
           f"descarga del repositorio público [{REPO}](https://github.com/{REPO}) sólo la carpeta "
           f"`{RUTA_EN_REPO}` de la edición `{EDICION}` (auxiliares, datos congelados y modelos). No pide acceso a Google Drive.")
NOTA_EN = ("> **Google Colab note**: the next cell installs `puremacro` and, if needed, fetches only the "
           f"`{RUTA_EN_REPO}` folder (helpers, frozen data and models) from the public repository "
           f"[{REPO}](https://github.com/{REPO}), edition `{EDICION}`. It does not ask for Google Drive access.")


def reescribir_celda0(fuente: str, nombre: str) -> str:
    es = _es_espanol(nombre)
    insignia = (f"[![{'Abrir en Colab' if es else 'Open In Colab'}]"
                f"(https://colab.research.google.com/assets/colab-badge.svg)]({_url_colab(nombre)})")
    nota = NOTA_ES if es else NOTA_EN
    if _INSIGNIA.search(fuente):
        fuente = _INSIGNIA.sub(lambda _m: insignia, fuente, count=1)
    else:
        fuente = insignia + "\n\n" + fuente
    if _NOTA_COLAB.search(fuente):
        fuente = _NOTA_COLAB.sub(lambda _m: nota, fuente, count=1)
    else:
        fuente = fuente.replace(insignia, insignia + "\n\n" + nota, 1)
    return fuente


# ---------------------------------------------------------------------------
# Celda 1: configuración del entorno
# ---------------------------------------------------------------------------
_PIP = re.compile(r"^\s*!pip install -q (.+)$", re.M)
_FIN_ESTILO = "    _nbstyle.apply_style()\nexcept ImportError:\n    pass"
_IMPORTS_BASE = {"import os", "import sys", "from pathlib import Path"}

PLANTILLA = '''# {setup_title}
import os
import sys
import subprocess
from pathlib import Path
{imports}
_course_requirements = {requirements!r}
_course_edition = {edition!r}

def _safe_probe(p, is_dir=None):
    """Test paths without raising in the Juno sandbox."""
    try:
        p = Path(p)
        if is_dir is True:
            return p.is_dir()
        if is_dir is False:
            return p.is_file()
        return p.exists()
    except (OSError, PermissionError, TypeError, ValueError):
        return False

if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_course_requirements], check=True)
    # A separate directory per edition prevents reuse of helpers from another course.
    _repo = Path("/content/puremacro-" + _course_edition)
    if not _safe_probe(_repo, is_dir=True):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                        "--branch", _course_edition, "https://github.com/{repo}.git", str(_repo)], check=True)
    subprocess.run(["git", "-C", str(_repo), "checkout", "--detach", _course_edition], check=True)
    subprocess.run(["git", "-C", str(_repo), "sparse-checkout", "set", "{ruta}"], check=True)
    _course_status = subprocess.run(["git", "-C", str(_repo), "status", "--porcelain",
                                     "--untracked-files=normal", "--", "{ruta}"],
                                    check=True, capture_output=True, text=True).stdout
    if _course_status.strip():
        raise RuntimeError("The cached course edition was modified; use a fresh Colab runtime for the frozen edition.")
    os.chdir(_repo / "{ruta}")

# Locate the course helpers for local Python, Colab and Juno.
_candidates = [Path.cwd(), Path.cwd() / "notebooks", Path.cwd() / "curso/notebooks"]
try:
    _candidates.insert(0, Path(__file__).resolve().parent)
except (NameError, OSError):
    pass
for _c in _candidates:
    if _safe_probe(_c / "_nbstyle.py", is_dir=False):
        if str(_c) not in sys.path:
            sys.path.insert(0, str(_c))
        break

import puremacro
if puremacro.__version__ != "5.5.0":
    raise RuntimeError("This course edition requires puremacro==5.5.0; install it and restart the kernel.")

# Cached reference responses are the reproducible default; live models are optional.
os.environ.setdefault("MAV_LLM_MODE", "cache")

%matplotlib inline
try:
    import _nbstyle
    _nbstyle.apply_style()
except ImportError:
    pass
'''


def reescribir_celda1(fuente: str, nombre: str) -> tuple[str, list[str]]:
    """Devuelve la celda de configuración nueva y una lista de avisos.

    De la celda maestra se conserva todo lo que no es localizar la carpeta: los
    paquetes extra del ``!pip install`` (p. ej. ``arch``), los ``import`` extra
    de la cabecera y cualquier código después de ``_nbstyle.apply_style()``.
    """
    avisos: list[str] = []
    if "_safe_probe" not in fuente or _FIN_ESTILO not in fuente:
        avisos.append("celda 1 no parece la de configuración conocida: se deja intacta")
        return fuente, avisos

    # Both legacy shell installs and this edition's structured list are inputs.
    m = re.search(r"^_course_requirements = (.+)$", fuente, re.M)
    if m:
        requisitos = ast.literal_eval(m.group(1))
    else:
        m = _PIP.search(fuente)
        requisitos = shlex.split(m.group(1)) if m else []
        if not m:
            avisos.append("celda 1 sin '!pip install': se añade el de puremacro")
    extras: set[str] = set()
    otros: list[str] = []
    for requisito in requisitos:
        paquete = re.match(r"^puremacro(?:\[([^]]+)\])?(?:[<>=!~].*)?$", requisito)
        if paquete:
            extras.update((paquete.group(1) or "").split(","))
        elif requisito not in otros:
            otros.append(requisito)
    extras.discard("")
    pin = PUREMACRO.replace("puremacro", "puremacro[" + ",".join(sorted(extras)) + "]", 1) if extras else PUREMACRO
    requisitos = [pin, *otros]

    cabecera = fuente.split("def _safe_probe", 1)[0]
    imports = []
    for node in ast.parse(cabecera).body:
        texto = ast.get_source_segment(cabecera, node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"_course_requirements", "_course_edition"}
            for target in node.targets
        ):
            continue
        if texto and texto not in _IMPORTS_BASE | {"import subprocess"}:
            # Retain conditional/multiline imports and any custom setup before
            # _safe_probe, as well as the trailing block below.
            imports.append(texto)

    cola = fuente.split(_FIN_ESTILO, 1)[1]

    # Preserve the whole cell when a replaced boilerplate block contains
    # unfamiliar imports, assignments or explicit custom path operations.
    avisos_previos = len(avisos)
    medio = fuente.split("def _safe_probe", 1)[1].split(_FIN_ESTILO, 1)[0]
    conocidas = ("drive.mount", "search_paths", "git", "colab_dir", "mav_target", "_candidates",
                 "colab", "clon", "sparse", "chdir", "_repo", "_candidatos", "MAV_LLM_MODE")
    for linea in medio.splitlines():
        s = linea.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        if re.match(r"^(import|from) ", s) and s not in ("import subprocess", "import puremacro", "import _nbstyle",
                                                          "from google.colab import drive"):
            avisos.append(f"celda 1: import no previsto en el bloque sustituido: {s}")
        if "sys.path.append" in s or ("os.environ" in s and not any(k in s for k in conocidas)):
            avisos.append(f"celda 1: línea no prevista en el bloque sustituido: {s}")

    medio_python = "def _safe_probe" + medio + _FIN_ESTILO
    medio_python = "\n".join(
        re.sub(r"^(\s*)[!%].*$", r"\1pass", line) for line in medio_python.splitlines()
    )
    variables_conocidas = {"p", "_repo", "_candidates", "_candidatos", "_c", "_course_status",
                           "search_paths", "colab_dir", "mav_target"}
    try:
        for node in ast.walk(ast.parse(medio_python)):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in variables_conocidas:
                        avisos.append("celda 1: asignación no prevista en el bloque sustituido: " + target.id)
    except SyntaxError:
        avisos.append("celda 1: bloque de configuración no reconocido")
    if len(avisos) > avisos_previos:
        return fuente, avisos

    titulo = ("Configuración: Google Colab, Python local o Juno (iPad)" if _es_espanol(nombre)
              else "Environment setup: Google Colab, local Python or Juno (iPad)")
    nueva = PLANTILLA.format(setup_title=titulo, imports="".join(ln + "\n" for ln in imports),
                            requirements=requisitos, edition=EDICION, repo=REPO, ruta=RUTA_EN_REPO).rstrip("\n")
    if cola.strip():
        nueva += "\n\n" + cola.strip("\n")
    return nueva, avisos


# ---------------------------------------------------------------------------
# 1. Cuadernos, auxiliares, datos y modelos
# ---------------------------------------------------------------------------
def cuadernos_fuente() -> list[Path]:
    return sorted(p for p in FUENTE_NB.glob("T*.ipynb") if p.is_file() and p.name not in NO_PUBLICAR_AUN)


def construir_cuadernos() -> dict:
    import nbformat

    conservar: set[Path] = set()
    escritos, avisos = 0, []
    codigo_total = []
    for src in cuadernos_fuente():
        nb = nbformat.read(str(src), as_version=4)
        celdas = nb.cells
        if len(celdas) < 2 or celdas[0].cell_type != "markdown" or celdas[1].cell_type != "code":
            avisos.append(f"{src.name}: las celdas 0-1 no son (markdown, código); no se reescriben")
        else:
            celdas[0].source = reescribir_celda0(celdas[0].source, src.name)
            celdas[1].source, av = reescribir_celda1(celdas[1].source, src.name)
            avisos += [f"{src.name}: {a}" for a in av]
        nb.cells = [c for c in nb.cells
                    if not (c.cell_type == "markdown" and c.source.lstrip().startswith(ENCABEZADOS_SOLUCIONES))]
        for viejo, nuevo in REESCRITURAS_INTERNAS.get(src.name, []):
            hits = [c for c in nb.cells if c.cell_type == "markdown" and viejo in c.source]
            if not hits:
                avisos.append(f"{src.name}: no aparece el texto interno a reescribir: {viejo[:60]!r}")
            for c in hits:
                c.source = c.source.replace(viejo, nuevo)
        codigo_total += [c.source for c in nb.cells if c.cell_type == "code"]
        texto_nb = nbformat.writes(nb)
        for local, publico in RUTAS_LOCALES:
            texto_nb = texto_nb.replace(local, publico)
        if "/Users/" in texto_nb:
            avisos.append(f"{src.name}: queda una ruta local en el cuaderno publicado")
        dst = DEST_NB / src.name
        escritos += _escribir_texto(dst, texto_nb)
        conservar.add(dst)

    for aux in AUXILIARES:
        dst = DEST_NB / aux
        escritos += _copiar_archivo(FUENTE_NB / aux, dst)
        conservar.add(dst)

    # data_curso: archivos reales
    datos = (FUENTE_NB / "data_curso").resolve()
    n_datos = 0
    if datos.is_dir():
        for src in sorted(datos.rglob("*")):
            rel = src.relative_to(datos)
            if any(parte in EXCLUIR_DATOS or parte.startswith(".") for parte in rel.parts) or rel in EXCLUIR_ARCHIVOS_DATOS:
                continue
            if src.is_file():
                dst = DEST_NB / "data_curso" / rel
                escritos += _copiar_archivo(src, dst)
                conservar.add(dst)
                n_datos += 1

    # modelos: los .mod y los *_results.mat que algún cuaderno nombra
    texto = "\n".join(codigo_total)
    modelos = FUENTE_NB / "modelos"
    fuera = []
    if modelos.is_dir():
        for src in sorted(modelos.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(modelos)
            if src.suffix == ".mod" and len(rel.parts) == 1:
                incluir = True
            elif src.suffix == ".mat" and src.name in texto:
                incluir = True
            else:
                incluir = False
                if src.suffix == ".mat":
                    fuera.append(str(rel))
            if incluir:
                dst = DEST_NB / "modelos" / rel
                escritos += _copiar_archivo(src, dst)
                conservar.add(dst)
    no_encontrados = sorted(set(re.findall(r"ruta_modelo\(\s*[\"']([^\"']+)[\"']", texto))
                            - {p.name for p in conservar} - {p.stem for p in conservar})
    if no_encontrados:
        avisos.append(f"modelos citados con ruta_modelo() que no se copiaron: {no_encontrados}")

    conservar |= {DEST_NB / "README.md", DEST_NB / "data_curso" / "FUENTES.md"}
    borrados = _podar(DEST_NB, conservar)
    return {"cuadernos": len(cuadernos_fuente()), "archivos_datos": n_datos, "escritos": escritos,
            "borrados": [str(b.relative_to(DESTINO)) for b in borrados],
            "mat_excluidos": fuera, "avisos": avisos}


# ---------------------------------------------------------------------------
# 2. Diapositivas
# ---------------------------------------------------------------------------
TITULOS_EN = {
    "Medición del ciclo: contabilidad nacional, filtros y hechos estilizados":
        "Measuring the cycle: national accounts, filters and stylized facts",
    "El modelo neoclásico de crecimiento": "The neoclassical growth model",
    "Riesgo e incertidumbre": "Risk and uncertainty",
    "Ciclos económicos reales y contabilidad del ciclo": "Real business cycles and business-cycle accounting",
    "Identificación empírica: SVAR, evidencia narrativa y proyecciones locales":
        "Empirical identification: SVARs, narrative evidence and local projections",
    "Mercados incompletos, heterogeneidad y HANK": "Incomplete markets, heterogeneity and HANK",
    "Mecanismos: trabajo, capital y economía pequeña y abierta":
        "Mechanisms: labour, capital and the small open economy",
    "Mercados no competitivos: precios rígidos y fricciones laborales":
        "Non-competitive markets: sticky prices and labour-market frictions",
}


def _argumento_llaves(tex: str, comando: str) -> str | None:
    """Contenido de ``\\comando{...}`` respetando llaves anidadas."""
    m = re.search(r"^[^%\n]*\\" + comando + r"\s*\{", tex, re.M)
    if not m:
        return None
    i, prof, out = m.end(), 1, []
    while i < len(tex) and prof:
        c = tex[i]
        prof += (c == "{") - (c == "}")
        if prof:
            out.append(c)
        i += 1
    return "".join(out)


def _limpiar_latex(s: str) -> str:
    s = re.sub(r"\\(?:large|Large|small|scriptsize|footnotesize|bf|it|em)\b", "", s)
    s = re.sub(r"\\text(?:bf|it)\{", "{", s)
    s = s.replace("\\\\", " ").replace("--", "–").replace("~", " ")
    s = re.sub(r"\\\[[^\]]*\]", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _slug(titulo: str) -> str:
    base = titulo.split(":")[0]
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode().lower()
    palabras = re.findall(r"[a-z0-9]+", base)
    if palabras and palabras[0] in {"el", "la", "los", "las"}:
        palabras = palabras[1:]
    return "_".join(palabras)


def construir_diapositivas() -> dict:
    try:
        from pypdf import PdfReader
    except ImportError:                                   # pragma: no cover
        PdfReader = None
    mazos, conservar, avisos, escritos = [], set(), [], 0
    fuente_diap = RAIZ
    if not (fuente_diap / "Slides01mav.pdf").is_file():
        alt = Path.home() / "Documents" / "TEACHING" / "MAV"
        if (alt / "Slides01mav.pdf").is_file():
            fuente_diap = alt
    for n in range(1, 9):
        pdf = fuente_diap / f"Slides0{n}mav.pdf"
        tex = fuente_diap / f"Slides0{n}mav.tex"
        if not pdf.is_file() or not tex.is_file():
            # Si no hay fuentes LaTeX, intenta usar los PDFs pre-compilados en site/diapositivas
            pre = RAIZ / "site" / "diapositivas"
            candidatos = list(pre.glob(f"MAV_{n:02d}_*.pdf")) if pre.is_dir() else []
            if candidatos:
                c_pdf = candidatos[0]
                dst = DEST_DIAP / c_pdf.name
                escritos += _copiar_archivo(c_pdf, dst)
                conservar.add(dst)
                paginas = len(PdfReader(str(c_pdf)).pages) if PdfReader else None
                mazos.append({
                    "numero": n, "archivo": c_pdf.name, "titulo_es": f"Mazo {n}", "titulo_en": f"Deck {n}",
                    "semanas": None, "bytes": c_pdf.stat().st_size, "paginas": paginas,
                })
                continue
            avisos.append(f"falta {pdf.name} o {tex.name}")
            continue
        fuente = tex.read_text(encoding="utf-8")
        titulo = _limpiar_latex(_argumento_llaves(fuente, "title") or f"Mazo {n}")
        subt = _limpiar_latex(_argumento_llaves(fuente, "subtitle") or "")
        m = re.search(r"Semanas? (\d+)(?:\s*–\s*(\d+))?", subt)
        semanas = (f"{m.group(1)}–{m.group(2)}" if m and m.group(2) else m.group(1)) if m else None
        titulo_en = TITULOS_EN.get(titulo)
        if titulo_en is None:
            avisos.append(f"mazo {n}: título sin traducción en TITULOS_EN: {titulo!r}")
        nombre = f"MAV_{n:02d}_{_slug(titulo)}.pdf"
        dst = DEST_DIAP / nombre
        escritos += _copiar_archivo(pdf, dst)
        conservar.add(dst)
        paginas = len(PdfReader(str(pdf)).pages) if PdfReader else None
        mazos.append({
            "numero": n, "archivo": nombre, "titulo_es": titulo, "titulo_en": titulo_en,
            "semanas": semanas, "bytes": pdf.stat().st_size, "paginas": paginas,
        })
    indice = {"curso": "Macroeconomía Avanzada (MAV), ITAM, otoño 2026", "mazos": mazos}
    ruta_indice = DEST_DIAP / "index.json"
    escritos += _escribir_texto(ruta_indice, json.dumps(indice, indent=2, ensure_ascii=False) + "\n")
    conservar.add(ruta_indice)
    # Sólo se podan PDFs viejos de este script; el resto de site/ no es nuestro.
    borrados = [p for p in DEST_DIAP.glob("MAV_*.pdf") if p not in conservar]
    for p in borrados:
        p.unlink()
    return {"mazos": mazos, "escritos": escritos, "borrados": [p.name for p in borrados], "avisos": avisos}


# ---------------------------------------------------------------------------
# 3. README y FUENTES
# ---------------------------------------------------------------------------
TEMAS = {  # código -> (tema en español, tema en inglés)
    "T00": ("Syllabus interactivo del curso", "Interactive course syllabus"),
    "T00_B": ("Configura tu modelo de lenguaje local (Qwen3-4B)", "Set up your local language model (Qwen3-4B)"),
    "T01_A": ("Hechos del ciclo económico desde el mercado de trabajo", "Business-cycle facts from the labour side"),
    "T01_B": ("Contabilidad nacional y hechos estilizados internacionales",
              "National accounts and cross-country stylized facts"),
    "T01_C": ("El ciclo a frecuencia mensual, semanal y diaria", "The cycle at monthly, weekly and daily frequency"),
    "T02_A": ("Modelo neoclásico de crecimiento: log-linealización y aproximación local",
              "Neoclassical growth model: log-linearization and local approximation"),
    "T02_B": ("RBC canónico: simulación, momentos y el debate de las horas",
              "Canonical RBC: simulation, moments and the hours debate"),
    "T02_C": ("Precios rígidos y búsqueda: fricciones en bienes y trabajo",
              "Sticky prices and search: frictions in goods and labour markets"),
    "T02_D": ("Restricciones ocasionalmente activas y límite inferior cero (OccBin)",
              "Occasionally binding constraints and the zero lower bound (OccBin)"),
    "T02_E": ("Estimación bayesiana de modelos DSGE y comparación de modelos",
              "Bayesian DSGE estimation and model comparison"),
    "T02_F": ("Fricciones macrofinancieras: Gertler y Karadi (2011)", "Macro-financial frictions: Gertler–Karadi (2011)"),
    "T03_A": ("Choques agregados con mercados incompletos: Krusell–Smith y jacobianos",
              "Aggregate shocks with incomplete markets: Krusell–Smith and sequence-space Jacobians"),
    "T03_B": ("Volatilidad: GARCH y efecto apalancamiento", "Volatility: GARCH and leverage effects"),
    "T03_C": ("Crecimiento en riesgo (growth-at-risk)", "Growth-at-risk"),
    "T03_D": ("Incertidumbre de política económica a partir de texto", "Economic policy uncertainty from text"),
    "T03_E": ("Modelos de lenguaje locales como instrumento de medición", "Local language models as measurement tools"),
    "T03_F": ("Comunicación de bancos centrales y tono de la política monetaria",
              "Central-bank communication and monetary-policy tone"),
    "T04_A": ("El multiplicador de impuestos, de tres maneras", "The tax multiplier, three ways"),
    "T04_C": ("Contabilidad del ciclo económico: las cuñas", "Business-cycle accounting: the wedges"),
    "T05_A": ("Identificación de choques con SVAR", "Identifying shocks with SVARs"),
    "T05_B": ("Proyecciones locales", "Local projections"),
    "T05_C": ("Diferencias en diferencias escalonadas", "Staggered difference-in-differences"),
    "T05_D": ("DiD con proyecciones locales (LP-DiD)", "Difference-in-differences with local projections (LP-DiD)"),
    "T05_E": ("Transmisión dependiente del estado y respuestas generalizadas",
              "State-dependent transmission and generalized impulse responses"),
    "T05_F": ("SVAR de la inflación de 2021–2024 (Bernanke–Blanchard)",
              "SVAR of the 2021–2024 inflation (Bernanke–Blanchard)"),
    "T05_G": ("Identificación narrativa de choques monetarios con un LLM local (Banxico y FOMC)",
              "Narrative identification of monetary shocks with a local LLM (Banxico and FOMC)"),
    "T06_A": ("¿De dónde viene la desigualdad de riqueza?", "Where does wealth inequality come from?"),
    "T06_B": ("Ciclo de vida, demografía y pensiones", "Life cycle, demographics and pensions"),
    "T06_C": ("Carteras, preferencias y la frontera HANK", "Portfolios, preferences and the HANK frontier"),
    "T07_A": ("Dinámica empresarial y mala asignación de recursos", "Firm dynamics and misallocation"),
    "T08_A": ("Búsqueda laboral y flujos de trabajadores", "Labour search and worker flows"),
    "T09_A": ("Galería de validación numérica", "Numerical validation gallery"),
    "T09_B": ("Construye tu propio índice", "Build your own index"),
}


def _pares_cuadernos() -> list[tuple[str, str | None, str | None]]:
    pares: dict[str, list] = {}
    for p in cuadernos_fuente():
        par = pares.setdefault(_codigo(p.name), [None, None])
        par[0 if _es_espanol(p.name) else 1] = p.name
    return [(c, es, en) for c, (es, en) in sorted(pares.items())]


def _enlace_colab(nombre: str | None, texto: str) -> str:
    return f"[{texto}]({_url_colab(nombre)})" if nombre else "—"


def construir_leeme() -> bool:
    filas_es, filas_en = [], []
    for codigo, es, en in _pares_cuadernos():
        tema_es, tema_en = TEMAS.get(codigo, (codigo, codigo))
        filas_es.append(f"| {codigo} | {tema_es} | {_enlace_colab(es, 'español')} | {_enlace_colab(en, 'inglés')} |")
        filas_en.append(f"| {codigo} | {tema_en} | {_enlace_colab(es, 'Spanish')} | {_enlace_colab(en, 'English')} |")
    tabla_es = "| Código | Tema | Español | Inglés |\n|---|---|---|---|\n" + "\n".join(filas_es)
    tabla_en = "| Code | Topic | Spanish | English |\n|---|---|---|---|\n" + "\n".join(filas_en)
    clon = (f"git clone --branch {EDICION} --depth 1 --filter=blob:none --sparse https://github.com/{REPO}.git\n"
            f"cd puremacro\ngit sparse-checkout set {RUTA_EN_REPO}\n"
            f"python -m pip install \"{PUREMACRO.replace('puremacro', 'puremacro[io]', 1)}\" statsmodels arch scikit-learn jupyterlab\n"
            f"cd {RUTA_EN_REPO}\njupyter lab")
    texto = f"""# Macroeconomía Avanzada (MAV) — cuadernos del curso

*English version below.*

Cuadernos de Jupyter del curso de Macroeconomía Avanzada (MAV) de la Maestría en
Economía del ITAM, otoño 2026. Todo el curso corre en Python con
[`puremacro`](https://github.com/{REPO}): cada cuaderno trae sus salidas ya
ejecutadas, así que se puede leer en GitHub sin correr nada.

La carpeta contiene los cuadernos en español (`*_es.ipynb`) y en inglés, los
auxiliares `_datos.py`, `_nbstyle.py`, `_tutor.py`, `_llm.py` y `_t01.py`, los datos congelados en
`data_curso/` y los modelos `.mod` en `modelos/`. Las diapositivas en PDF están
en `curso/site/diapositivas/`.

## Abrir un cuaderno en Google Colab

Haga clic en el enlace del cuaderno. La primera celda de código instala
`puremacro` y descarga sólo esta carpeta del repositorio (un clon parcial de los materiales
y datos congelados); no pide acceso a Google Drive. Ejecute las celdas en orden.

{tabla_es}

## Correr los cuadernos en su máquina

Requiere Python 3.12 o posterior.

```bash
{clon}
```

La edición fija `puremacro==5.5.0` y el contenido del tag `{EDICION}`: auxiliares,
datos, caché y modelos. `puremacro.dp` y `puremacro.text` son obligatorios y sus
secciones se ejecutan completas. Los 72 cuadernos incluyen las variantes históricas
de T01; la tabla muestra la ruta principal.

La instalación inicial requiere red. Después se usan los datos congelados de
`data_curso/` y, por omisión, `MAV_LLM_MODE=cache`: respuestas de referencia de
Qwen3-4B, sin buscar motores ni descargar modelos. T00_B también corre en ese modo;
la comparación con un modelo vivo y los refrescos de datos son opcionales.
Para usar un modelo local, siga T00_B (`pip install "puremacro[local-llm]==5.5.0"`
u Ollama) y elija explícitamente `MAV_LLM_MODE=auto`.

## Datos

Los archivos de `data_curso/` son copias congeladas de datos públicos de
terceros (FRED, OCDE, INEGI, ILOSTAT, Banco Mundial y otros). La licencia MIT de
`puremacro` cubre el código, **no** esos datos. La procedencia de cada archivo y
cómo citarlo están en [`data_curso/FUENTES.md`](data_curso/FUENTES.md). Si usa
estos datos, cite a los editores originales.

## Licencia

El código es MIT. El texto de los cuadernos y las diapositivas del curso están bajo
[Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.es):
puede reutilizarlos y adaptarlos, también con fines comerciales, citando
«Jorge Alonso-Ortiz, *Macroeconomía Avanzada*, ITAM». Los datos conservan los
términos de sus editores.

---

# Advanced Macroeconomics (MAV) — course notebooks

Jupyter notebooks for the Advanced Macroeconomics (MAV) course in ITAM's Master's
programme in Economics, fall 2026. The whole course runs in Python with
[`puremacro`](https://github.com/{REPO}). Every notebook is saved with its
executed outputs, so you can read it on GitHub without running anything.

This folder holds the notebooks in Spanish (`*_es.ipynb`) and English, the
helpers `_datos.py`, `_nbstyle.py`, `_tutor.py`, `_llm.py` and `_t01.py`, the frozen data in
`data_curso/` and the `.mod` models in `modelos/`. The PDF slides are in
`curso/site/diapositivas/` (in Spanish).

## Open a notebook in Google Colab

Click the notebook's link. The first code cell installs `puremacro` and fetches
only this folder from the repository (a partial clone of the materials
and frozen data); it does not ask for Google Drive access. Run the cells in order.

{tabla_en}

## Run the notebooks locally

Requires Python 3.12 or later.

```bash
{clon}
```

This edition pins `puremacro==5.5.0` and tag `{EDICION}` for its helpers, data,
cache and models. `puremacro.dp` and `puremacro.text` are required and their
sections execute in full. All 72 notebooks include the historical T01 variants;
the table shows the main route.

Initial installation needs the network. Afterwards, notebooks use frozen files
in `data_curso/` and default to `MAV_LLM_MODE=cache`: Qwen3-4B reference responses,
without engine discovery or model downloads. T00_B also runs in this mode;
comparison with a live model and live data refreshes are optional.
For a local model, follow T00_B (`pip install "puremacro[local-llm]==5.5.0"` or
Ollama) and explicitly select `MAV_LLM_MODE=auto`.

## Data

The files in `data_curso/` are frozen copies of public third-party data (FRED,
OECD, INEGI, ILOSTAT, World Bank and others). The MIT licence of `puremacro`
covers the code, **not** that data. Where each file comes from, and how to cite
it, is in [`data_curso/FUENTES.md`](data_curso/FUENTES.md). If you use this data,
cite the original publishers.

## Licence

The code is MIT. The notebook text and the course slides are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):
you may reuse and adapt them, commercially too, crediting
"Jorge Alonso-Ortiz, *Macroeconomía Avanzada*, ITAM". The data keep their publishers' terms.
"""
    return _escribir_texto(DEST_NB / "README.md", texto)


# Editor original de las series que llegan por FRED.
_FRED_EDITOR = [
    (r"^[A-Z]{2}UR$", "bls"),
    (r"^(A191RL1Q225SBEA|GDPA|GDPC1|GPDI|GPDIC1|PCECC96|PNFI|Y001RC1Q027SBEA)$", "bea"),
    (r"^(AWHI|CE16OV|CLF16OV|CNP16OV|CPIAUCSL|CPILFESL|HOANBS|JTSHIL|JTSJOL|LEU\d+Q|LN[SU]\d+|OPHNFB|OUTNFB|"
     r"PRS\d+|UEMPLT5|UNEMPLOY|UNRATE|USPRIV)$", "bls"),
    (r"^(FEDFUNDS|INDPRO|TCU|NCBEILQ027S|TNWMVBSNNCB|WFRBS[A-Z]\d+)$", "fed"),
    (r"^NFCI$", "chicagofed"),
    (r"^USREC$", "nber"),
    (r"^VIXCLS$", "cboe"),
    (r"^(CPALTT01MXM659N|CPGRLE01MXM659N|IR3TIB01MXM156N|LRUN64TTMXQ156S)$", "ocde"),
    (r"^POPTOT\w+WDB$", "bancomundial"),
    (r"^\w+RUG$", "pwt"),
]

EDITORES = {
    "bea": ("U.S. Bureau of Economic Analysis (BEA), vía FRED", "https://www.bea.gov",
            "Obra del gobierno federal de EUA. Cite a BEA y a FRED (Federal Reserve Bank of St. Louis)."),
    "bls": ("U.S. Bureau of Labor Statistics (BLS), vía FRED", "https://www.bls.gov",
            "Obra del gobierno federal de EUA. Cite a BLS y a FRED. Incluye CPS, JOLTS, LAUS, productividad."),
    "fed": ("Board of Governors of the Federal Reserve System, vía FRED", "https://www.federalreserve.gov",
            "Incluye G.17, Z.1 (Financial Accounts) y las Distributional Financial Accounts. Cite a la Junta de "
            "la Reserva Federal y a FRED."),
    "chicagofed": ("Federal Reserve Bank of Chicago (NFCI), vía FRED", "https://www.chicagofed.org",
                   "Consulte los términos del editor. Cite al Chicago Fed."),
    "nber": ("NBER (fechas de recesión), vía FRED", "https://www.nber.org/research/business-cycle-dating",
             "Consulte los términos del editor. Cite al NBER Business Cycle Dating Committee."),
    "cboe": ("Cboe Global Markets (VIX), vía FRED", "https://www.cboe.com",
             "Serie con derechos de un tercero: consulte los términos de Cboe antes de redistribuirla."),
    "fred_otras": ("Otras series de FRED", "https://fred.stlouisfed.org",
                   "FRED redistribuye series de muchos editores y algunas tienen derechos de terceros: "
                   "consulte la ficha de cada serie en FRED y los términos de su editor."),
    "ocde": ("OCDE (QNA, STES, MEI, ANA, KEI, LFS)", "https://data-explorer.oecd.org",
             "Reutilización permitida con atribución según los términos de la OCDE; consúltelos. Cite a la OCDE, "
             "no a este curso. Varios archivos son paneles derivados (rebases, empalmes) construidos con "
             "puremacro.fetch a partir de estos datos."),
    "bancomundial": ("Banco Mundial (World Development Indicators)", "https://data.worldbank.org",
                     "Licencia CC BY 4.0. Cite al Banco Mundial."),
    "pwt": ("Penn World Table (Universidad de Groningen), vía FRED", "https://www.rug.nl/ggdc/productivity/pwt/",
            "Licencia CC BY 4.0. Cite Feenstra, Inklaar y Timmer (2015)."),
    "inegi": ("INEGI — ENOE (procesada)", "https://www.inegi.org.mx/programas/enoe/",
              "Consulte los Términos de libre uso del INEGI. Cite al INEGI. Los archivos son agregados procesados, "
              "no microdatos."),
    "ilo": ("OIT — ILOSTAT", "https://ilostat.ilo.org", "Consulte los términos del editor. Cite a ILOSTAT (OIT)."),
    "eurostat": ("Eurostat (vacantes y paro), con JOLTS y UNRATE para EUA", "https://ec.europa.eu/eurostat",
                 "Consulte los términos de Eurostat y de BLS. Cite a ambos."),
    "epu": ("Economic Policy Uncertainty — policyuncertainty.com", "https://www.policyuncertainty.com",
            "Consulte los términos del sitio. Cite Baker, Bloom y Davis (2016)."),
    "fernald": ("John Fernald — PTF trimestral ajustada por utilización (Federal Reserve Bank of San Francisco)",
                "https://www.frbsf.org/research-and-insights/data-and-indicators/total-factor-productivity-tfp/",
                "Consulte los términos del editor. Cite Fernald (2014)."),
    "philfed": ("Federal Reserve Bank of Philadelphia — Real-Time Data Set for Macroeconomists (y ALFRED)",
                "https://www.philadelphiafed.org/surveys-and-data/real-time-data-research",
                "Consulte los términos del editor. Cite a la Reserva Federal de Filadelfia."),
    "replicas": ("Archivos de réplica de artículos (Ramey–Zubairy 2018, Romer–Romer 2010, Galí 1999, "
                 "Devries–Guajardo–Leigh–Pescatori 2011)", "",
                 "Consulte los términos de cada archivo de réplica. Cite el artículo original. Galí (1999) se "
                 "reconstruye con series de BLS en FRED."),
    "jln": ("Jurado, Ludvigson y Ng — incertidumbre macro", "https://www.sydneyludvigson.com",
            "Consulte los términos del editor. Cite Jurado, Ludvigson y Ng (2015)."),
    "klems": ("EU KLEMS", "https://euklems-intanprod-llee.luiss.it",
              "Consulte los términos del editor. Cite la base EU KLEMS."),
    "banxico": ("Banco de México — anuncios de política monetaria, clasificados para el curso",
                "https://www.banxico.org.mx",
                "Clasificación propia del curso a partir de comunicados públicos de Banxico. Cite a Banxico."),
    "curso": ("Cálculos del curso", "",
              "Resultados calculados con puremacro (Aiyagari, contabilidad del ciclo) a partir de los datos de "
              "arriba. Cite las fuentes de origen."),
}


def _editor(archivo: str, fuente: str) -> str:
    serie = archivo.split(".")[0]
    f = fuente.lower()
    if f.startswith("fred") or "snapshot puremacro.replication (fred" in f:
        if "replication" in f:
            return "replicas"
        if "laus" in f or "cps" in f:
            return "bls"
        for patron, clave in _FRED_EDITOR:
            if re.match(patron, serie):
                return clave
        return "fred_otras"
    reglas = [("salida de t01_b", "ocde"), ("alfred", "philfed"), ("philadelphia", "philfed"), ("filadelfia", "philfed"),
              ("enoe", "inegi"), ("ilostat", "ilo"), ("eurostat", "eurostat"), ("policyuncertainty", "epu"),
              ("fernald", "fernald"), ("replication", "replicas"), ("devries", "replicas"),
              ("jurado", "jln"), ("klems", "klems"), ("banxico", "banxico"), ("banco mundial", "bancomundial"),
              ("computo local", "curso"), ("contabilidad del ciclo", "curso"), ("ocde", "ocde"),
              ("qna_long_panel", "ocde"), ("cps", "bls"), ("beige", "fed")]
    for clave_txt, clave in reglas:
        if clave_txt in f:
            return clave
    if archivo.startswith("oecd_"):
        return "ocde"
    if archivo.startswith("banxico_"):
        return "banxico"
    if archivo.startswith("fomc_"):
        return "fed"
    if archivo.startswith("wedges_"):
        return "curso"
    return "fred_otras" if re.match(r"^[A-Z0-9]+$", serie) else "curso"


def construir_fuentes() -> bool:
    manifiesto = (FUENTE_NB / "data_curso").resolve() / "bundle_2026A" / "manifest.csv"
    filas = {}
    if manifiesto.is_file():
        with manifiesto.open(encoding="utf-8") as fh:
            filas = {r["archivo"]: r for r in csv.DictReader(fh)}
    presentes = sorted({p.name for p in (DEST_NB / "data_curso").rglob("*") if p.is_file()}
                       - {"FUENTES.md", "LEEME.md", "manifest.csv", "manifest.json"})
    grupos: dict[str, list[str]] = {}
    for a in presentes:
        fuente = filas.get(a, {}).get("fuente", "")
        grupos.setdefault(_editor(a, fuente), []).append(a)

    bloques = []
    for clave, (nombre, url, terminos) in EDITORES.items():
        if clave not in grupos:
            continue
        cab = f"### {nombre}\n\n" + (f"<{url}>\n\n" if url else "") + terminos + "\n\n"
        lista = ", ".join(f"`{a}`" for a in grupos[clave])
        bloques.append(cab + lista + "\n")
    texto = f"""# Fuentes de `data_curso/`

*English summary at the end.*

Estos archivos son **copias congeladas** (paquete `bundle_2026A`, agosto de 2026)
de datos públicos de terceros, más algunos cálculos del curso hechos a partir de
ellos. Se incluyen para que los cuadernos corran sin red y den siempre las mismas
cifras. `bundle_2026A/manifest.csv` registra, archivo por archivo, la fuente, la
fecha de descarga, la última observación y el SHA-256. Los archivos de la raíz de
`data_curso/` son idénticos a los de `bundle_2026A/`.

**La licencia MIT de `puremacro` cubre su código, no estos datos.** Cada editor
conserva sus derechos y sus términos de uso. Donde abajo no se indica una licencia
concreta, consulte los términos del editor antes de redistribuir. **Si usa estos
series cambian después de publicadas (revisiones): para un trabajo propio,
descargue la versión vigente de la fuente.

## Por editor

{chr(10).join(bloques)}
---

## English summary

These files are frozen copies (bundle `bundle_2026A`, August 2026) of public
third-party data, plus a few course computations derived from them, so the
notebooks run offline and reproduce the same numbers. `bundle_2026A/manifest.csv`
records each file's source, download date, last observation and SHA-256.

**The MIT licence of `puremacro` covers its code, not this data.** Each publisher
keeps its rights and terms of use; the licence is stated above only where it is
known (World Bank and Penn World Table: CC BY 4.0; U.S. federal agencies: U.S.
government works). Otherwise, check the publisher's terms before redistributing.
**If you use this data, cite the original publishers** listed above, not this
course or `puremacro`. No publisher is affiliated with or endorses this course.
"""
    return _escribir_texto(DEST_NB / "data_curso" / "FUENTES.md", texto)


# ---------------------------------------------------------------------------
# 4. Tamaños
# ---------------------------------------------------------------------------
def resumen_tamanos(carpetas: list[Path]) -> dict:
    archivos = [p for c in carpetas if c.is_dir() for p in c.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in archivos)
    grandes = sorted(archivos, key=lambda p: p.stat().st_size, reverse=True)[:8]
    return {
        "archivos": len(archivos), "total_mb": round(total / MB, 1),
        "mayores": [(str(p.relative_to(DESTINO)), round(p.stat().st_size / MB, 2)) for p in grandes],
        "sobre_50mb": [str(p.relative_to(DESTINO)) for p in archivos if p.stat().st_size > AVISO_MB * MB],
        "sobre_100mb": [str(p.relative_to(DESTINO)) for p in archivos if p.stat().st_size > LIMITE_MB * MB],
    }


def main(argv: list[str] | None = None) -> int:
    global DESTINO, DEST_NB, DEST_DIAP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", action="append", choices=["cuadernos", "diapositivas", "leeme"],
                    help="construir sólo esta parte (se puede repetir)")
    ap.add_argument("--destino", type=Path, default=None, help="directorio de destino")
    a = ap.parse_args(argv)
    if a.destino:
        DESTINO = a.destino.resolve()
        DEST_NB = DESTINO / "notebooks"
        DEST_DIAP = DESTINO / "site" / "diapositivas"

    partes = set(a.solo or ["cuadernos", "diapositivas", "leeme"])

    if "cuadernos" in partes:
        r = construir_cuadernos()
        print(f"cuadernos: {r['cuadernos']}  archivos de datos: {r['archivos_datos']}  "
              f"escritos: {r['escritos']}  borrados: {len(r['borrados'])}")
        for b in r["borrados"]:
            print(f"   borrado: {b}")
        if r["mat_excluidos"]:
            print(f"   .mat no citados (fuera): {r['mat_excluidos']}")
        for av in r["avisos"]:
            print(f"   AVISO {av}")
    if "diapositivas" in partes:
        r = construir_diapositivas()
        print(f"diapositivas: {len(r['mazos'])} mazos  escritos: {r['escritos']}")
        for m in r["mazos"]:
            print(f"   {m['archivo']:62s} {m['paginas']:>4} pág  {m['bytes'] / MB:5.2f} MB  semanas {m['semanas']}")
        for av in r["avisos"]:
            print(f"   AVISO {av}")
    if "leeme" in partes:
        e1, e2 = construir_leeme(), construir_fuentes()
        print(f"README.md {'escrito' if e1 else 'sin cambios'};  FUENTES.md {'escrito' if e2 else 'sin cambios'}")

    t = resumen_tamanos([DEST_NB, DEST_DIAP])
    print(f"\ntamaño: {t['archivos']} archivos, {t['total_mb']} MB (notebooks/ + site/diapositivas/)")
    for ruta, mb in t["mayores"][:5]:
        print(f"   {mb:7.2f} MB  {ruta}")
    for ruta in t["sobre_50mb"]:
        print(f"   AVISO más de {AVISO_MB} MB (GitHub advierte): {ruta}")
    for ruta in t["sobre_100mb"]:
        print(f"   ERROR más de {LIMITE_MB} MB (GitHub lo rechaza): {ruta}")
    return 1 if t["sobre_100mb"] else 0


if __name__ == "__main__":
    sys.exit(main())
