
## Update 2026-06-22 — Docker won't run in this environment

Installed Docker 29.1.3 successfully, but **containers cannot run here**:
- Container has Docker's default caps — no `NET_ADMIN`/`SYS_ADMIN` → no bridge
  networking, can't create namespaces.
- seccomp blocks `unshare` → rootless Docker impossible, and even image-layer
  extraction fails (`failed to register layer: unshare: operation not permitted`).
- Can't change the outer container's seccomp/caps from inside it.

So `sandbox=("docker", "compose.yaml")` can't execute end-to-end here. The
docker scaffolding is still correct for a host that has privileges.

**To run end-to-end, two things are needed (open question for human):**
1. A run path: either (a) a local-sandbox variant — run `pet_api` + a MailHog
   substitute as host processes, `sandbox="local"`, addresses → localhost; or
   (b) run the existing docker setup on a privileged host/VM.
2. A model API key (none set: ANTHROPIC/OPENAI/etc. all unset; no local ollama).
