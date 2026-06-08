<script lang="ts">
  import { Download, Eye, Table2 } from 'lucide-svelte';
  import { downloadBytes } from '$lib/utils/files';

  export let title: string;
  export let url = '';
  export let filename: string;
  export let bytes: Uint8Array | undefined = undefined;
  export let onOpen: () => void = () => undefined;
</script>

<section class="image-card">
  <div class="image-card-header">
    <h3>{title}</h3>
    <div class="floating-actions">
      {#if url}
        <button class="icon-button" on:click={onOpen} title="Open image"><Eye size={16} /></button>
        {#if bytes}<button class="icon-button" on:click={() => downloadBytes(bytes, filename)} title="Download image"><Download size={16} /></button>{/if}
      {/if}
    </div>
  </div>
  {#if url}
    <button class="image-button" on:click={onOpen} aria-label={`Open ${title}`}><img src={url} alt={title} /></button>
  {:else}
    <div class="image-placeholder"><Table2 size={24} /> Waiting</div>
  {/if}
</section>
