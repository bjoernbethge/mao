"""
Storage-related API endpoints (Configuration and Import/Export).
"""

import os
import json
import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import FileResponse

from .db import ConfigDB
from .models import Config
from .api import get_config_db

# Create router
config_router = APIRouter(prefix="/config", tags=["config"])
export_router = APIRouter(tags=["import-export"])

# Global configuration endpoints
@config_router.post("", status_code=204)
def set_config_value(config: Config, db: ConfigDB = Depends(get_config_db)):
    """Sets a global configuration value"""
    db.set_config(config.key, config.value, config.description)
    return None

@config_router.get("/{key}")
def get_config_value(key: str, db: ConfigDB = Depends(get_config_db)):
    """Gets a global configuration value by its key"""
    value = db.get_config(key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Configuration key {key} not found")
    return {"key": key, "value": value}

@config_router.get("")
def list_all_configs(db: ConfigDB = Depends(get_config_db)):
    """Lists all global configuration values"""
    conn = db._connect()
    results = conn.execute("SELECT * FROM global_configs").fetchall()
    
    configs = []
    for row in results:
        config = dict(zip(row.keys(), row))
        if config.get('value'):
            config['value'] = json.loads(config['value'])
        configs.append(config)
    
    return configs

@config_router.delete("/{key}", status_code=204)
def delete_config_value(key: str, db: ConfigDB = Depends(get_config_db)):
    """Deletes a global configuration value"""
    db.delete_config(key)
    return None

# Import/Export endpoints
@export_router.post("/export", status_code=200)
def export_configuration(db: ConfigDB = Depends(get_config_db)):
    """Exports the entire configuration as JSON"""
    export_path = db.export_config()
    
    # Datei als JSON zurückgeben
    with open(export_path, 'r') as f:
        config_data = json.load(f)
    
    return config_data

@export_router.get("/export-file")
def export_configuration_file(db: ConfigDB = Depends(get_config_db)):
    """Exportiert die gesamte Konfiguration als JSON-Datei zum Download"""
    export_path = db.export_config(export_path="mcp_config_export.json")
    
    # Datei zum Download anbieten
    return FileResponse(
        path=export_path, 
        filename="mcp_config_export.json", 
        media_type="application/json"
    )

@export_router.post("/import", status_code=204)
async def import_configuration(
    config_data: Dict[str, Any],
    db: ConfigDB = Depends(get_config_db)
):
    """Importiert eine Konfiguration aus JSON-Daten"""
    # Temporäre Datei für den Import erstellen
    import_path = f"temp_import_{uuid.uuid4().hex}.json"
    
    try:
        with open(import_path, 'w') as f:
            json.dump(config_data, f)
        
        success = db.import_config(import_path)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to import configuration")
        
        return None
    finally:
        # Temporäre Datei entfernen
        if os.path.exists(import_path):
            os.remove(import_path)
            
@export_router.post("/import-file", status_code=204)
async def import_configuration_file(
    file: UploadFile = File(...),
    db: ConfigDB = Depends(get_config_db)
):
    """Importiert eine Konfiguration aus einer hochgeladenen JSON-Datei"""
    # Temporäre Datei für den Import erstellen
    import_path = f"temp_import_{uuid.uuid4().hex}.json"
    
    try:
        # Datei speichern
        content = await file.read()
        with open(import_path, 'wb') as f:
            f.write(content)
        
        # Konfiguration importieren
        success = db.import_config(import_path)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to import configuration")
        
        return None
    finally:
        # Temporäre Datei entfernen
        if os.path.exists(import_path):
            os.remove(import_path)
            
@export_router.post("/reset", status_code=204)
def reset_database(confirm: bool = False, db: ConfigDB = Depends(get_config_db)):
    """Setzt die Datenbank zurück (löscht alle Daten)"""
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required to reset database")
    
    conn = db._connect()
    
    # Alle Tabellen leeren
    tables = [
        "agents", 
        "tools", 
        "mcp_servers", 
        "agent_tools", 
        "global_configs"
    ]
    
    for table in tables:
        conn.execute(f"DELETE FROM {table}")
    
    return None 