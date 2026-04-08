"""API key authentication for WebSocket connections."""

import http
import logging

import websockets.asyncio.server as _server

from hosting.config import CustomerId
from hosting.config import HasAuth

logger = logging.getLogger(__name__)

# Maps connection id -> customer_id after successful auth.
# Populated in process_request, consumed (popped) in the handler.
# Uses id(connection) which is unique for the lifetime of the connection object.
_authenticated_connections: dict[int, CustomerId] = {}


def pop_customer_id(connection: _server.ServerConnection) -> CustomerId | None:
    """Retrieve and remove the customer_id for an authenticated connection.

    Returns None if the connection was not authenticated (should not happen
    if process_request rejected unauthenticated upgrades).
    """
    return _authenticated_connections.pop(id(connection), None)


def create_request_handler(service_config: HasAuth):
    """Create a process_request callback that checks API keys and handles health checks.

    Returns a function compatible with websockets' process_request parameter.
    The returned function:
    - Serves /healthz with 200 OK
    - Checks Authorization: Bearer <key> header on all other requests
    - Rejects unauthorized requests with 401
    - Returns None to continue with normal WebSocket handling
    """

    def process_request(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
        # Health check endpoint — no auth required.
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")

        # Extract API key from Authorization header.
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning("Rejected connection: missing or malformed Authorization header")
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Missing Bearer token\n")

        api_key = auth_header.removeprefix("Bearer ")
        customer = service_config.lookup_api_key(api_key)
        if customer is None:
            logger.warning("Rejected connection: invalid API key")
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Invalid API key\n")

        # Track customer_id by connection id for the handler to read.
        _authenticated_connections[id(connection)] = customer.customer_id
        logger.info("Authenticated connection for customer=%s", customer.customer_id)

        # Continue with WebSocket upgrade.
        return None

    return process_request
