// pm2 app for scripts/lfm_serve_8114.sh (LFM K3.5 on GPU 2): restarts only on crash.
module.exports = {
  apps: [{
    name: "lfm-k35",
    script: __dirname + "/lfm_serve_8114.sh",
    interpreter: "bash",
    cwd: __dirname + "/..",
    autorestart: true,
    max_restarts: 10,
    min_uptime: "180s",
    kill_timeout: 15000,
  }],
};
