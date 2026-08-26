"""保存 SQLite 产品数据库的版本和建表语句。"""

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS participants (
    participant_id TEXT PRIMARY KEY,
    display_name TEXT,
    jersey_color TEXT,
    jersey_number TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_participants_jersey
    ON participants (jersey_color, jersey_number);

CREATE TABLE IF NOT EXISTS participant_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    embedding_json TEXT NOT NULL,
    model_name TEXT,
    source_track_id TEXT,
    quality_score REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (participant_id)
        REFERENCES participants (participant_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_participant
    ON participant_embeddings (participant_id);

CREATE TABLE IF NOT EXISTS materials (
    material_id TEXT PRIMARY KEY,
    source_video_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    video_path TEXT NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    processing_status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (end_seconds >= start_seconds),
    UNIQUE (source_video_id, segment_id)
);

CREATE INDEX IF NOT EXISTS idx_materials_source_time
    ON materials (source_video_id, start_seconds);

CREATE TABLE IF NOT EXISTS material_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT NOT NULL,
    event_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    player_id TEXT,
    start_seconds REAL,
    end_seconds REAL,
    FOREIGN KEY (material_id)
        REFERENCES materials (material_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_material_events_lookup
    ON material_events (event_name, confidence);

CREATE TABLE IF NOT EXISTS material_participants (
    material_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    jersey_color TEXT,
    jersey_number TEXT,
    player_name TEXT,
    identity_status TEXT,
    reid_cluster_id TEXT,
    PRIMARY KEY (material_id, participant_id, track_id),
    FOREIGN KEY (material_id)
        REFERENCES materials (material_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_material_participants_lookup
    ON material_participants (participant_id);
"""
