// pm2 app for scripts/truss_serve_8113.sh: restarts only on crash, and when RSS passes 64 GB (the host budget beside
// the renters; RSS counts the mmap'd model's file pages too, ~42 GB peak measured in TRACKER #85).
module.exports = {
  apps: [{
    name: "truss-8113",
    script: __dirname + "/truss_serve_8113.sh",
    interpreter: "bash",
    cwd: __dirname + "/..",
    autorestart: true,
    max_restarts: 3,
    min_uptime: "120s",
    max_memory_restart: "64G",
    kill_timeout: 15000,
  }],
};
