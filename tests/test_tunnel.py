"""Smoke test for the Modal tunnel deployment.

Reads the tunnel address from Modal Dict and runs benchmark inferences.
"""

from typing import cast

from modal import Dict
from openpi_client import websocket_client_policy as _websocket_client_policy

from hosting.benchmark import InferablePolicy, run_benchmark
from hosting.modal_dict_names import OPENPI_MODAL_TUNNEL_INFO_DICT_NAME
from tests.helpers import random_observation_aloha, wait_for_server


def run() -> None:
    print("Reading tunnel address from Modal Dict ...")
    tunnel_dict = Dict.from_name(OPENPI_MODAL_TUNNEL_INFO_DICT_NAME)
    ws_url = tunnel_dict["url"]
    print(f"Tunnel address: {ws_url}")

    health_url = ws_url.replace("wss://", "https://").replace("ws://", "http://") + "/healthz"
    wait_for_server(health_url)

    print(f"\nConnecting to {ws_url} ...")
    policy = _websocket_client_policy.WebsocketClientPolicy(host=ws_url)
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(cast(InferablePolicy, policy), random_observation_aloha)
    result.print_summary()
