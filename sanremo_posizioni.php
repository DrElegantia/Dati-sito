<?php

declare(strict_types=1);

$dashboardPath = __DIR__ . '/docs/sanremo_dashboard.json';
$error = null;
$records = [];

if (!is_readable($dashboardPath)) {
    $error = 'Il file dati di Sanremo non è disponibile.';
} else {
    $rawJson = file_get_contents($dashboardPath);
    $decoded = json_decode($rawJson ?: '', true);

    if (!is_array($decoded)) {
        $error = 'Il file JSON di Sanremo non è valido.';
    } else {
        $performances = $decoded['performances'] ?? [];

        foreach ($performances as $item) {
            if (!is_array($item)) {
                continue;
            }

            $records[] = [
                'year' => $item['year'] ?? null,
                'serata' => $item['serata'] ?? null,
                'artist' => $item['artist'] ?? '-',
                'position' => $item['performance_order'] ?? null,
                'total' => $item['total_in_serata'] ?? null,
                'televoto' => $item['classifica_serata_televoto'] ?? null,
                'complessiva' => $item['classifica_serata_complessiva'] ?? null,
            ];
        }

        usort($records, static function (array $a, array $b): int {
            return [$b['year'], $a['serata'], $a['position']] <=> [$a['year'], $b['serata'], $b['position']];
        });
    }
}
?>
<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sanremo posizioni</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }
    h1 { margin-bottom: 0.25rem; }
    .hint { color: #94a3b8; margin-top: 0; }
    .error { background: #7f1d1d; color: #fee2e2; padding: 0.75rem; border-radius: 0.5rem; }
    table { border-collapse: collapse; width: 100%; background: #111827; margin-top: 1rem; }
    th, td { padding: 0.55rem; border-bottom: 1px solid #1f2937; text-align: left; }
    th { background: #1f2937; }
  </style>
</head>
<body>
  <h1>Posizioni esibizioni Sanremo</h1>
  <p class="hint">Vista PHP costruita da <code>docs/sanremo_dashboard.json</code></p>

  <?php if ($error !== null): ?>
    <div class="error"><?= htmlspecialchars($error, ENT_QUOTES, 'UTF-8'); ?></div>
  <?php else: ?>
    <table>
      <thead>
      <tr>
        <th>Anno</th>
        <th>Serata</th>
        <th>Artista</th>
        <th>Posizione</th>
        <th>Totale artisti</th>
        <th>Classifica televoto</th>
        <th>Classifica complessiva</th>
      </tr>
      </thead>
      <tbody>
      <?php foreach ($records as $row): ?>
        <tr>
          <td><?= htmlspecialchars((string) ($row['year'] ?? '-'), ENT_QUOTES, 'UTF-8'); ?></td>
          <td><?= htmlspecialchars((string) ($row['serata'] ?? '-'), ENT_QUOTES, 'UTF-8'); ?></td>
          <td><?= htmlspecialchars((string) ($row['artist'] ?? '-'), ENT_QUOTES, 'UTF-8'); ?></td>
          <td><?= htmlspecialchars((string) ($row['position'] ?? '-'), ENT_QUOTES, 'UTF-8'); ?></td>
          <td><?= htmlspecialchars((string) ($row['total'] ?? '-'), ENT_QUOTES, 'UTF-8'); ?></td>
          <td><?= htmlspecialchars((string) ($row['televoto'] ?? '-'), ENT_QUOTES, 'UTF-8'); ?></td>
          <td><?= htmlspecialchars((string) ($row['complessiva'] ?? '-'), ENT_QUOTES, 'UTF-8'); ?></td>
        </tr>
      <?php endforeach; ?>
      </tbody>
    </table>
  <?php endif; ?>
</body>
</html>
