import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()
ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "conversation_log.csv"
LOG_COLUMNS = ["actor", "message", "tool_call", "timestamp"]
DEFAULT_API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Configuración del cliente LLM (Groq)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
)

# Modelo activo y estable para Groq
MODEL = "openai/gpt-oss-120b"

# System prompt mejorado para evitar bucles
SYSTEM_PROMPT = """Eres el asistente de inventario de Carla.
Usa la API de inventario como única fuente de verdad; nunca inventes IDs ni stocks.
Si el usuario pregunta por el inventario general, usa SIEMPRE la herramienta 'get_inventory'.
¡IMPORTANTE! No llames a la misma herramienta varias veces seguidas. Si ya obtuviste la información, úsala para responder.
No conviertas unidades ni pronostiques demanda semanal.
Si una herramienta falla, infórmalo honestamente."""

def tool_schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }

DECIMAL = {"type": ["number", "string"], "description": "Valor numérico o decimal en la unidad fija del producto"}

TOOLS = [
    tool_schema("get_inventory", "Lista todos los productos con sus IDs, cantidades y unidades", {}, []),
    tool_schema("add_product", "Registra un nuevo producto con su unidad explícita", {
        "name": {"type": "string"},
        "quantity": DECIMAL,
        "unit": {"type": "string"}
    }, ["name", "quantity", "unit"]),
    tool_schema("change_stock", "Aplica un delta con signo a un producto existente (positivo para entradas, negativo para salidas)", {
        "product_id": {"type": "integer", "minimum": 1},
        "delta": DECIMAL
    }, ["product_id", "delta"]),
    tool_schema("get_alerts", "Lista productos estrictamente por debajo del umbral de alerta", {
        "threshold": DECIMAL
    }, []),
]

def log_event(actor: str, message: str, tool_call: str = ""):
    file_exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "actor": actor,
            "message": str(message),
            "tool_call": tool_call,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

def execute_tool(name, arguments, api_base_url=DEFAULT_API):
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        method, route, body = "GET", "/inventory", None
        
        if name == "get_inventory":
            pass  # Usa los valores por defecto (GET /inventory)
        elif name == "add_product":
            method, body = "POST", args
        elif name == "change_stock":
            method = "PATCH"
            route = f"/inventory/{args['product_id']}"
            body = {"delta": args["delta"]}
        elif name == "get_alerts":
            route += "/alerts?" + urlencode({"threshold": args.get("threshold", 10)})
        else:
            raise ValueError("Herramienta desconocida")

        url = api_base_url.rstrip("/") + route
        req = Request(
            url,
            method=method,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def run_agent_loop():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("🤖 Agente de Inventario iniciado (escribe 'salir' para terminar)\n")

    while True:
        user_input = input("Tú: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            break

        log_event("user", user_input)
        messages.append({"role": "user", "content": user_input})

        # Bucle interno de Observar → Pensar → Actuar → Actualizar
        while True:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except OpenAIError as e:
                print(f"❌ Error del LLM: {e}")
                break

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                reply = message.content or ""
                print(f"\nCarla: {reply}\n")
                log_event("agent", reply)
                break

            for tool_call in message.tool_calls:
                print(f"⚙️ [Ejecutando tool: {tool_call.function.name}]")
                log_event("agent", "Llamada a herramienta", tool_call.function.name)

                result = execute_tool(tool_call.function.name, tool_call.function.arguments)
                result_str = json.dumps(result, ensure_ascii=False)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })
                log_event("tool", result_str, tool_call.function.name)

if __name__ == "__main__":
    run_agent_loop()