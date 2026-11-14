import csv
from decimal import Decimal, localcontext
import os
from pathlib import Path
from threading import Lock
from typing import Annotated
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, StringConstraints

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / os.environ.get("INVENTORY_CSV_PATH", "products.csv")
HEADER = ["id", "name", "quantity", "unit"]
LOCK = Lock()

app = FastAPI(title="Inventory API - Carla")

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Stock = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]

class NewProduct(BaseModel):
    name: Text
    quantity: Stock
    unit: Text

class Product(NewProduct):
    id: Annotated[int, Field(gt=0)]

class StockChange(BaseModel):
    delta: FiniteDecimal

def read_products() -> list[Product]:
    # Verifica si el archivo no existe o está vacío para evitar errores
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return []
    
    try:
        with CSV_PATH.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source, strict=True)
            if reader.fieldnames != HEADER:
                raise ValueError("Cabecera CSV incorrecta")
            
            products = []
            ids = set()
            for row in reader:
                if set(row) != set(HEADER) or any(v is None for v in row.values()):
                    raise ValueError("Fila CSV inválida")
                
                product = Product.model_validate(row)
                if product.id in ids:
                    raise ValueError("ID duplicado")
                
                ids.add(product.id)
                products.append(product)
            return products
            
    except (ValueError, csv.Error, OSError) as e:
        raise HTTPException(500, "Error al leer el archivo CSV") from e

def write_products(products: list[Product]) -> None:
    try:
        with CSV_PATH.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(p.model_dump() for p in products)
    except OSError as e:
        raise HTTPException(500, "Error al escribir el archivo CSV") from e

@app.get("/inventory", response_model=list[Product])
def inventory():
    with LOCK:
        return read_products()

@app.post("/inventory", response_model=Product, status_code=201)
def add_product(body: NewProduct):
    with LOCK:
        products = read_products()
        new_id = max((p.id for p in products), default=0) + 1
        product = Product(id=new_id, **body.model_dump())
        products.append(product)
        write_products(products)
        return product

@app.patch("/inventory/{product_id}", response_model=Product)
def change_stock(product_id: int, body: StockChange):
    with LOCK:
        products = read_products()
        for product in products:
            if product.id == product_id:
                with localcontext() as context:
                    context.prec = 28
                    new_qty = product.quantity + body.delta
                    if new_qty < 0:
                        raise HTTPException(400, "El stock resultante no puede ser negativo")
                    product.quantity = new_qty
                write_products(products)
                return product
        raise HTTPException(404, f"Producto {product_id} no encontrado")

@app.get("/inventory/alerts", response_model=list[Product])
def alerts(threshold: Annotated[Stock, Query()] = Decimal("10")):
    with LOCK:
        return [p for p in read_products() if p.quantity < threshold]