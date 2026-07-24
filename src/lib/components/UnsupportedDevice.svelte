<script lang="ts">
  import { Monitor } from 'lucide-svelte';
</script>

<div class="unsupported-shell">
  <section class="unsupported-card panel">
    <div class="unsupported-icon">
      <Monitor size={28} />
    </div>
    <p class="eyebrow">Retinalysis</p>
    <h1>Desktop only</h1>
    <p class="unsupported-copy">
      VascX Browser runs on-device AI models and needs a desktop PC or Mac with enough memory
      and a full browser. Phones and tablets are not supported.
    </p>
    <p class="unsupported-hint">Please open this page on a desktop computer.</p>
  </section>
</div>
