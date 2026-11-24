# news-sentiment-analyzer: análisis rápido (24-nov-2026)

Diagnóstico, sin cambios al código. Pedido por Giovanni.

## Veredicto

El proyecto está sano para ser de portafolio: **61 de 61 pruebas pasan** (Python 3.13 local, offline, 6.5 s; CI cubre 3.11 y 3.12), el README es honesto con sus límites y el endurecimiento de seguridad está bien hecho. Lo que le falta es de **producto**, no de calidad: el CSV no guarda fecha ni descripción, así que hoy sirve para mirar un corte, no para alimentar a otro sistema.

## Contexto consultado

- CLAUDE.md y README: coherentes con el código (backends, umbrales ±0.05, desempates de tema, comportamiento ante fallos de Claude).
- Graphiti (grupos del repo y global): sólo hechos de arquitectura del 3-nov (stack, archivos, PR #3 de CLAUDE.md). Nada sobre decisiones pendientes.
- Qdrant: nada específico de este repo. Sí hay una nota de sked del mismo día (ver sección "Nemotron").

## Revisión de Nemotron (OpenCode)

**No se pudo.** `opencode_run` respondió `Unrecognized flag: --dir in command opencode run` incluso con un mensaje de una línea, antes de mandar código. Es el mismo fallo que la nota de sked registró hoy (dos intentos). El error está en cómo el conector invoca la CLI, no en el mensaje. Por eso no hay nada que comparar contra Nemotron; la revisión de abajo es mía. Nota: la herramienta anuncia el modelo vía OpenRouter, pero como no corrió, no se envió código a ningún servicio externo.

Cuando se arregle el conector (quitar `--dir` o actualizar la CLI), conviene repetir esto y anotar qué hallazgos coinciden.

## Hallazgos del código (revisión propia)

Ordenados por utilidad. Ninguno rompe hoy.

1. **Se pierde la fecha de publicación.** `Article.published` se parsea pero `analyze_batch` no lo pasa a `results.csv` (`src/sentiment.py`, diccionario final). Tampoco salen `description` ni un id estable. Es lo que más limita usarlo como alimentador.
2. **El tono se calcula sobre título + descripción** (`Article.full_text`), pero README y prompt de Claude dicen "headline". El README sí menciona "headlines y summaries" en Limitaciones; conviene unificar la redacción para no prometer de más.
3. **Enlace de Atom.** `parse_feed` toma el primer `<link>` sin mirar `rel`; en feeds que antepongan `rel="self"` el enlace sería el del feed, no el del artículo. Ningún feed configurado es Atom, así que es latente.
4. **`urllib` acepta cualquier esquema** (`file://`, `ftp://`) si alguien pasa un `--feeds` ajeno. Riesgo bajo (el archivo lo controla quien ejecuta), pero basta validar `http/https` en `_http_get`, igual que ya se hace con los enlaces del reporte.
5. **Descarga secuencial, timeout de 8 s por operación de socket** (no total) y sin reintentos: un feed lento retrasa toda la corrida. Aceptable para CLI; mejorable si se usa programado.
6. **Sin deduplicación entre feeds ni entre corridas.** Relevante en cuanto se ejecute periódicamente.
7. **Tests:** el único aviso es un `PendingDeprecationWarning` de seaborn (`set_bad`); inocuo.
8. **User-Agent fijo** (`news-sentiment-analyzer/1.1`): si algún medio empieza a bloquear por UA, el síntoma sería un feed "caído" en el aviso. No lo observé, pero Firecrawl usa navegador y no lo habría detectado (ver abajo).

Lo bueno, para el registro: defusedxml, límite de 5 MB, guarda de fórmulas en CSV (`'` inicial), escape de Markdown/HTML y enlaces sólo http/https, fallback por lote, código de salida 3 para errores de petición.

## Verificación

- `pip install -r requirements.txt` y `pytest -q`: **61 passed, 1 warning.**
- **Feeds vivos** (Firecrawl, 24-nov-2026, `maxAge: 0`, estado 200 y ítems presentes; fechas del 24-nov confirmadas sólo en Folha y El Financiero): BBC World (27 ítems), The Guardian World (45), El Universal (57), El Financiero (100), La Jornada (99), Expansión (36), Folha de S.Paulo (titulares del 24-nov), g1 (100). **8 de 11 comprobados.** No comprobé BBC Technology, BBC Mundo ni BBC News Brasil (mismo host que BBC World, que sí responde).
  - Limitación: Firecrawl entrega la página renderizada por un navegador, no el XML crudo ni con el User-Agent del proyecto. Confirma que el feed existe y tiene contenido; no que `parse_feed` lo lea idéntico ni que no haya bloqueo por UA.
  - El Financiero es el más pesado (~2 MB renderizado por Firecrawl; el XML crudo no lo medí): conviene vigilarlo frente al límite de 5 MB.
- **`DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"`:** coincide con el identificador de Haiku 5.5 de la familia Claude 5 vigente. **No hice una llamada real** (no hay `ANTHROPIC_API_KEY` en esta sesión ni se deben usar claves en tests), así que "vigente" está contrastado contra la lista de modelos, no contra la API.

## ¿Sirve como alimentador de sked o del newsfeeder de hibrida?

**Sked: sí, como pieza, no como está.** Sked arma la escaleta y la pondera con audiencia; de este repo le sirven `feeds.py` (lectura RSS/Atom endurecida, bilingüe es/pt/en) y el clasificador de temas. El tono sirve menos para una escaleta (un titular negativo no es "peor" para el noticiario). Para enchufarlo faltan: fecha y descripción en la salida, un id estable, salida JSON/JSONL y deduplicación. Sería un cambio chico (hallazgo 1 y 6). La nota de sked indica que `alimentadores/` debe tener una sola fuente de verdad en su repo; yo lo copiaría como módulo ahí o lo importaría, no mantendría dos parsers.

**Newsfeeder de hibrida: no puedo opinar con base.** No tengo acceso al repo de hibrida ni encontré en memoria un newsfeeder concreto. Lo único cercano: el plan del módulo de "señales" de KORE (esquema normalizado plataforma / id / texto / métricas / fecha, hashear autor, guardar texto y métricas, no identidades). Titulares de medios con fecha y tono encajan en ese esquema como una fuente más; con los mismos cambios de arriba. Para decidir si el newsfeeder existente ya hace lo mismo, habría que ver su código.

## Conectores (sin activar)

1. **Firecrawl: vigilar fuentes.** Un monitor de Firecrawl sobre los feeds de `config/feeds.json` avisaría cuando uno deje de responder o cambie de forma (hoy sólo se entera uno por el `[warn]` de una corrida). También sirve para medios sin RSS vía `firecrawl_scrape` con salida JSON (Apify no lo verifiqué en esta sesión).
2. **Slack: alerta de tono.** Resumen diario o alerta cuando la proporción de titulares negativos de una fuente o tema pase un umbral, enviando el `report.md` o su resumen a un canal. Requiere ejecutar el análisis en una tarea programada y decidir canal; no lo activé.

## Próximos pasos propuestos (decide Giovanni)

1. Agregar `published`, `description` e id estable a `results.csv`, más `--format jsonl` (pruebas incluidas, README actualizado).
2. Validar esquema http/https en `_http_get` y elegir el `<link>` Atom con `rel="alternate"`.
3. Arreglar el conector OpenCode y repetir la revisión con Nemotron para comparar.
