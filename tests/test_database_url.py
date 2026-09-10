from app.config import build_database_urls


def test_database_url_uses_psycopg_drivers_and_ssl() -> None:
    sync_url, async_url = build_database_urls(
        "postgresql://xirvo:secret@dpg-example.render.com:5432/xirvo_closing_agent"
    )
    assert sync_url.drivername == "postgresql+psycopg"
    assert async_url.drivername == "postgresql+psycopg_async"
    assert sync_url.host == "dpg-example.render.com"
    assert sync_url.database == "xirvo_closing_agent"
    assert sync_url.query["sslmode"] == "require"
    assert async_url.query["sslmode"] == "require"


def test_postgres_scheme_is_normalized() -> None:
    sync_url, _async_url = build_database_urls(
        "postgres://xirvo:secret@dpg-example.render.com:5432/xirvo_closing_agent"
    )
    assert sync_url.drivername == "postgresql+psycopg"


def test_existing_sslmode_is_preserved() -> None:
    sync_url, _async_url = build_database_urls(
        "postgresql://xirvo:secret@localhost:5432/xirvo_rag?sslmode=prefer"
    )
    assert sync_url.query["sslmode"] == "prefer"


def test_split_local_fields_do_not_force_ssl() -> None:
    sync_url, async_url = build_database_urls(
        None,
        db_host="localhost",
        db_port=5432,
        db_name="Xirvo_Closing_Agent",
        db_user="postgres",
        db_password="local",
    )
    assert sync_url.drivername == "postgresql+psycopg"
    assert async_url.drivername == "postgresql+psycopg_async"
    assert "sslmode" not in sync_url.query
    assert sync_url.host == "localhost"
    assert sync_url.database == "Xirvo_Closing_Agent"
