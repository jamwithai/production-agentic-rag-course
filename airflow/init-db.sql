-- Initialize additional databases on first PostgreSQL startup
-- This runs automatically via /docker-entrypoint-initdb.d/
CREATE DATABASE airflow_db;