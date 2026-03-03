<?php
http_response_code(404);
?>
<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>404 - Pagina non trovata</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0f172a; color: #e2e8f0; }
    .card { max-width: 42rem; padding: 2rem; border-radius: 1rem; background: #111827; box-shadow: 0 20px 45px rgba(0,0,0,.4); }
    h1 { margin-top: 0; font-size: 2rem; }
    p { color: #94a3b8; }
    ul { list-style: none; padding: 0; margin: 1.25rem 0 0; display: flex; gap: 0.75rem; flex-wrap: wrap; }
    a { display: inline-block; text-decoration: none; color: #0f172a; background: #93c5fd; padding: 0.55rem 0.8rem; border-radius: 0.55rem; font-weight: 700; }
    a:hover { background: #bfdbfe; }
  </style>
</head>
<body>
  <main class="card">
    <h1>404 — Pagina non trovata</h1>
    <p>La risorsa richiesta non è disponibile. Puoi tornare rapidamente alle sezioni principali del sito:</p>
    <ul>
      <li><a href="/macro">Macro</a></li>
      <li><a href="/articoli">Articoli</a></li>
      <li><a href="/">Home</a></li>
    </ul>
  </main>
</body>
</html>
