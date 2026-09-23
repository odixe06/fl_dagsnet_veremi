#!/usr/bin/env python
"""Helper the credential-rotation wizard calls once per account. Never prints a token.

    rotate_kaggle_creds.py install-kgat USER      # new KGAT on stdin: verify, install, revoke old
    rotate_kaggle_creds.py rotate-oauth USER      # after `kaggle auth login --force` as USER:
                                                  #   snapshot the new login, revoke the old refresh token
    rotate_kaggle_creds.py resync-hosts [DIR...]  # rewrite host MCP configs that carry a replaced KGAT
    rotate_kaggle_creds.py status                 # which accounts still carry a *.prev file

Files under ~/.kaggle/accounts/ (mode 600):
    USER.mcp-token         current KGAT bearer            USER.mcp-token.prev        replaced value
    USER.credentials.json  current OAuth snapshot         USER.credentials.json.prev replaced snapshot
The .prev files exist only between a rotation and `resync-hosts`, which deletes them.
"""
import json, os, shutil, sys
from pathlib import Path

ACC = Path.home() / ".kaggle" / "accounts"
LIVE = Path.home() / ".kaggle" / "credentials.json"
REASON = "rotated by rotate_kaggle_creds.py"


def _write600(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def introspect_kgat(tok):
    """(active, username) for a KGAT bearer, from the server."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
    api = KaggleApi.build_kaggle_client_with_params(args=[], api_token=tok)
    req = IntrospectTokenRequest()
    req.token = tok
    with api:
        r = api.security.oauth_client.introspect_token(req)
    return bool(r.active), r.username or ""


def expire_token(auth_tok, victim):
    """POST /api/v1/tokens/revoke for `victim`, authenticated with `auth_tok` (a KGAT or access token)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.users.types.account_service import ExpireApiTokenRequest
    api = KaggleApi.build_kaggle_client_with_params(args=[], api_token=auth_tok)
    req = ExpireApiTokenRequest()
    req.token = victim
    req.reason = REASON
    with api:
        api.users.account_client.expire_api_token(req)


def snapshot_creds(path):
    """Load an OAuth snapshot without touching the live login; returns (creds, client)."""
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    client = KaggleClient()
    creds = KaggleCredentials.load(client, file_path=str(path))
    if creds is None:
        raise RuntimeError(f"{path.name}: no usable refresh token")
    return creds, client


def cmd_install_kgat(user):
    new = sys.stdin.read().strip().removeprefix("Bearer ").strip()
    if not new.startswith("KGAT_") or any(c.isspace() for c in new):
        print("  ✗ that is not a KGAT_ token; nothing changed")
        return 1
    active, who = introspect_kgat(new)
    if not active or who != user:
        print(f"  ✗ server says active={active} username={who or '?'}; expected {user}. Nothing changed.")
        return 1
    print(f"  ✓ new token introspects active as {user}")
    tokf = ACC / f"{user}.mcp-token"
    old = tokf.read_text().strip().removeprefix("Bearer ").strip() if tokf.is_file() else ""
    if old == new:
        print("  = same value as the one already stored; nothing to rotate")
        return 0
    ACC.mkdir(parents=True, exist_ok=True)
    os.chmod(ACC, 0o700)
    if old:
        _write600(ACC / f"{user}.mcp-token.prev", old + "\n")
    _write600(tokf, new + "\n")
    print(f"  ✓ stored as {tokf.name} (mode 600)")
    if not old:
        return 0
    try:
        o_active, _ = introspect_kgat(old)
    except Exception as exc:
        print(f"  ? could not introspect the old token ({type(exc).__name__}); treating it as live")
        o_active = True
    if not o_active:
        print("  ✓ old token is already inactive on the server")
        return 0
    try:
        expire_token(new, old)
        o_active, _ = introspect_kgat(old)
        print("  ✓ old token revoked via API" if not o_active
              else "  ⚠ revoke call returned but the old token still introspects ACTIVE — delete it on the website")
    except Exception as exc:
        print(f"  ⚠ revoke via API failed ({type(exc).__name__}) — delete the old token on the website")
    return 0


def cmd_rotate_oauth(user):
    try:
        live = json.loads(LIVE.read_text())
    except (OSError, ValueError):
        print("  ✗ no live login in ~/.kaggle/credentials.json; nothing changed")
        return 1
    if live.get("username") != user:
        print(f"  ✗ the live login is '{live.get('username')}', not '{user}'. Nothing changed.")
        return 1
    creds, client = snapshot_creds(LIVE)
    with client:
        creds._access_token = creds.generate_access_token().token
        who = creds.introspect()
    if who != user:
        print(f"  ✗ server says the live login belongs to {who}; nothing changed")
        return 1
    print(f"  ✓ new login introspects as {user}")

    snap = ACC / f"{user}.credentials.json"
    old_rt = ""
    if snap.is_file():
        old_rt = json.loads(snap.read_text()).get("refresh_token", "")
        if old_rt == live.get("refresh_token"):
            print("  = the snapshot already holds this refresh token; nothing to rotate")
            return 0
        shutil.copy2(snap, ACC / f"{user}.credentials.json.prev")
        os.chmod(ACC / f"{user}.credentials.json.prev", 0o600)
    ACC.mkdir(parents=True, exist_ok=True)
    os.chmod(ACC, 0o700)
    shutil.copy2(LIVE, snap)
    os.chmod(snap, 0o600)
    print(f"  ✓ snapshot {snap.name} replaced (mode 600)")
    if not old_rt:
        return 0

    # Revoke the old refresh token the way `kaggle auth revoke` does — mint an access token
    # from it, then POST it to /tokens/revoke — but WITHOUT KaggleCredentials.revoke_token(),
    # whose delete() removes ~/.kaggle/credentials.json (the new live login) regardless of
    # which file the credentials were loaded from. A dead token fails at the mint step.
    prev = ACC / f"{user}.credentials.json.prev"
    ocreds, oclient = snapshot_creds(prev)
    with oclient:
        try:
            old_at = ocreds.generate_access_token().token
        except Exception as exc:
            print(f"  ✓ old refresh token no longer mints a token ({type(exc).__name__}); nothing to revoke")
            return 0
    try:
        expire_token(old_at, old_rt)
    except Exception as exc:
        print(f"  ⚠ revoke via API failed ({type(exc).__name__}) — revoke the old grant on the website")
        return 0
    ocreds, oclient = snapshot_creds(prev)
    with oclient:
        try:
            ocreds.generate_access_token()
            print("  ⚠ revoke returned but the old refresh token STILL mints a token — revoke it on the website")
        except Exception:
            print("  ✓ old refresh token revoked (it no longer mints an access token)")
    return 0


def cmd_resync_hosts(dirs):
    prev = {p.name.removesuffix(".mcp-token.prev"): p.read_text().strip().removeprefix("Bearer ").strip()
            for p in ACC.glob("*.mcp-token.prev")}
    cur = {u: (ACC / f"{u}.mcp-token").read_text().strip().removeprefix("Bearer ").strip() for u in prev}
    roots = [Path(d).expanduser() for d in dirs] or [Path.cwd()]
    n = 0
    for root in roots:
        for f in list(root.rglob(".mcp.json")) + list(root.rglob(".vscode/mcp.json")):
            if any(part in ("node_modules", ".git") for part in f.parts):
                continue
            txt = f.read_text()
            hit = [u for u, t in prev.items() if t in txt]
            if hit:
                for u in hit:
                    txt = txt.replace(prev[u], cur[u])
                _write600(f, txt)
                n += 1
                print(f"  ✓ {f}: bearer of {', '.join(hit)} replaced (mode 600)")
            elif f.stat().st_mode & 0o077:
                os.chmod(f, 0o600)
                print(f"  ✓ {f}: mode tightened to 600")
    print(f"  {n} host config(s) rewritten")
    for p in list(ACC.glob("*.mcp-token.prev")) + list(ACC.glob("*.credentials.json.prev")):
        p.unlink()
    print("  ✓ .prev files removed")
    return 0


def cmd_status():
    for p in sorted(ACC.glob("*.prev")):
        print(f"  pending: {p.name}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "install-kgat":
        sys.exit(cmd_install_kgat(rest[0]))
    if cmd == "rotate-oauth":
        sys.exit(cmd_rotate_oauth(rest[0]))
    if cmd == "resync-hosts":
        sys.exit(cmd_resync_hosts(rest))
    if cmd == "status":
        sys.exit(cmd_status())
    sys.exit(__doc__)
