# Real-time inference for robots at Physical Intelligence

*Customer Stories • April 8, 2026 • 5 minute read*

[Physical Intelligence](https://www.physicalintelligence.company/) (Pi) is building a general-purpose robotic intelligence system capable of operating any robot across any task. Their core model---a Visual-Language-Action (VLA) architecture---takes visual observations, natural-language instructions, and the robot's proprioceptive state, then outputs motor commands for the next fraction of a second. Every arm movement in their system flows through this closed loop of continuous inference.

To evaluate progress, Pi doesn't just rely on simulation. Every model revision must be validated on real robots performing real tasks. That means thousands of inference cycles running 24/7 across a growing fleet of robots.

Running this compute on Modal simplified operations and enabled rapid experimentation with larger models, while only adding 10-15ms of network overhead.

## Designing for the constraints of real-time latency

For standard local inference, Pi models are trained in the cloud then downloaded locally to run using on-board GPUs attached to each robotic system. This provides a reliable system with good performance and straightforward debugging, but requires a GPU on each robot.

Off-board remote inference provided a more lightweight solution for robots without an onboard GPU.

### Architecting low-latency applications on Modal

For most Modal customers running latency-sensitive services, the solution is [Modal Tunnels](https://modal.com/docs/guide/tunnels). Tunnels expose live TCP ports on a running Modal container directly to the public internet, with automatic TLS termination and a secure, randomly assigned URL. Within less than a minute of container startup time, a service running inside the container is reachable over HTTPS or raw TCP.

With Tunnels, you start a container, call `modal.forward(port)`, and Modal handles the routing, TLS, and connection management. They're fast, direct, and reliable, which is why they're commonly used for interactive workloads like live APIs, Jupyter notebooks, VS Code servers, and real-time dashboards. All Modal features---GPUs, attached volumes, custom images---work seamlessly behind a tunnel and are typically the right primitive for applications needing direct container access.

However, because Pi's robots sit inside a continuous control loop, even small amounts of jitter or head-of-line blocking can degrade behavior, which means the nature of TCP was not necessarily the right architectural choice for the setup.

To match Tunnels-like performance gains without introducing the potential for request-response stalls in the control loop, Pi worked with Modal to build a more specialized transport: **a QUIC-based portal running over UDP with automatic NAT traversal**.

This system establishes a persistent, bidirectional channel between the robot runtime and the Modal GPU container. Instead of issuing independent TCP requests, the robot connects once and streams observations while receiving action outputs over the same channel.

The portal handles automatic NAT traversal using built-in STUN discovery and UDP hole punching, coordinated via [Modal Dict](https://modal.com/docs/guide/dicts) for rendezvous. This lets robots connect directly to Modal containers---even when both sides are behind NAT---without Pi needing to run custom relay infrastructure. Once the hole-punched path is established, communication runs over QUIC, implemented in Rust for high throughput and minimal latency.

This approach meant the cloud only added roughly ~10-15 ms of network overhead.

On Modal, Pi can allocate larger, data-center-class GPUs per deployment and run GPU-intensive experiments immediately. This lets the team experiment with larger models that couldn't even fit on the on-board GPUs, so engineers can test ambitious architectures first and optimize later, if needed.

With Modal, Pi also moved checkpoints into [Modal Volumes](https://modal.com/docs/guide/volumes) and mounts them directly into GPU containers. Model state now lives alongside the compute that consumes it and takes less than 30 seconds to load checkpoints.

## Expanding global reach

Even the best protocols can't overcome the speed of light, so where the GPU runs matters.

Modal lets Pi pin inference deployments to specific regions close to the robots, keeping the round-trip path short and predictable.

Once inference is region-pinned, adding robots in a new location stops being an infrastructure build-out. Pi can bring up the same container near the new site, mount the same checkpoints via Modal Volumes, and connect robots over the same low-latency transport---without shipping GPU towers or standing up a local cluster.

---

*Source: [Modal Blog](https://modal.com/blog/physical-intelligence-runs-real-time-remote-inference-for-robotic-control-on-modal)*
