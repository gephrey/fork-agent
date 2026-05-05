<script lang="ts">
  import { onMount } from 'svelte';

  // State
  let theme = $state('');
  let mood = $state('creative');
  let intensity = $state(70);
  let filter = $state<'all' | 'favorites'>('all');
  let isGenerating = $state(false);
  let favorites = $state<Set<number>>(new Set());
  let cards = $state<Array<{
    id: number;
    title: string;
    prompt: string;
    colorTags: string[];
    heat: number;
  }>>([]);

  // Load from localStorage on mount
  onMount(() => {
    const savedTheme = localStorage.getItem('moodboard-theme');
    const savedMood = localStorage.getItem('moodboard-mood') || 'creative';
    const savedIntensity = localStorage.getItem('moodboard-intensity');
    const savedFavorites = localStorage.getItem('moodboard-favorites');
    const savedCards = localStorage.getItem('moodboard-cards');

    if (savedTheme) theme = savedTheme;
    if (savedIntensity) intensity = parseInt(savedIntensity, 10);
    mood = savedMood as any;

    if (savedFavorites) {
      try {
        const favs = JSON.parse(savedFavorites);
        favorites = new Set(favs);
      } catch (e) {
        console.warn('Failed to parse favorites', e);
      }
    }

    if (savedCards) {
      try {
        cards = JSON.parse(savedCards);
      } catch (e) {
        console.warn('Failed to parse cards', e);
      }
    }
  });

  // Save to localStorage whenever state changes
  $effect(() => {
    localStorage.setItem('moodboard-theme', theme);
    localStorage.setItem('moodboard-mood', mood);
    localStorage.setItem('moodboard-intensity', intensity.toString());
    localStorage.setItem('moodboard-favorites', JSON.stringify(Array.from(favorites)));
    localStorage.setItem('moodboard-cards', JSON.stringify(cards));
  });

  // Mock AI generation logic
  function generateCards() {
    if (!theme.trim()) return;
    isGenerating = true;

    // Simulate API delay
    setTimeout(() => {
      const newCards = Array.from({ length: 6 }, (_, i) => {
        const baseTitle = [
          'Neon Dreamscape',
          'Quantum Pulse',
          'Cyber Zen Garden',
          'Holographic Mirage',
          'Synthwave Horizon',
          'Void Bloom'
        ][i % 6];

        const prompts = [
          `A futuristic ${theme} interface with glowing neon accents and fluid animations, ultra-detailed, cinematic lighting`,
          `Minimalist ${theme} concept art in a dark ambient setting, soft gradients, subtle motion blur`,
          `${theme} reimagined as a bioluminescent ecosystem, deep blues and violets, ethereal glow`,
          `Retro-futuristic ${theme} poster with chrome textures, grid overlays, and chromatic aberration`,
          `Abstract ${theme} composition using fractal geometry and particle systems, high contrast`,
          `${theme} as a sentient nebula — cosmic dust, iridescent sheen, volumetric light`
        ];

        const colors = [
          ['neon-cyan', 'midnight-blue', 'electric-purple'],
          ['slate-gray', 'amber-glow', 'smoke-white'],
          ['deep-indigo', 'bioluminescent-green', 'void-black'],
          ['chrome-silver', 'vintage-red', 'grid-gray'],
          ['obsidian', 'plasma-orange', 'quantum-teal'],
          ['cosmic-purple', 'iridescent-pink', 'stardust-white']
        ];

        const heats = [89, 92, 85, 94, 87, 91];

        return {
          id: Date.now() + i,
          title: baseTitle,
          prompt: prompts[i % 6],
          colorTags: colors[i % 6],
          heat: heats[i % 6]
        };
      });

      cards = newCards;
      isGenerating = false;
    }, 1200);
  }

  function toggleFavorite(id: number) {
    if (favorites.has(id)) {
      favorites.delete(id);
    } else {
      favorites.add(id);
    }
  }

  function refreshCard(id: number) {
    const idx = cards.findIndex(c => c.id === id);
    if (idx === -1) return;

    const baseTitle = [
      'Neon Dreamscape',
      'Quantum Pulse',
      'Cyber Zen Garden',
      'Holographic Mirage',
      'Synthwave Horizon',
      'Void Bloom'
    ][idx % 6];

    const prompts = [
      `A futuristic ${theme} interface with glowing neon accents and fluid animations, ultra-detailed, cinematic lighting`,
      `Minimalist ${theme} concept art in a dark ambient setting, soft gradients, subtle motion blur`,
      `${theme} reimagined as a bioluminescent ecosystem, deep blues and violets, ethereal glow`,
      `Retro-futuristic ${theme} poster with chrome textures, grid overlays, and chromatic aberration`,
      `Abstract ${theme} composition using fractal geometry and particle systems, high contrast`,
      `${theme} as a sentient nebula — cosmic dust, iridescent sheen, volumetric light`
    ];

    const colors = [
      ['neon-cyan', 'midnight-blue', 'electric-purple'],
      ['slate-gray', 'amber-glow', 'smoke-white'],
      ['deep-indigo', 'bioluminescent-green', 'void-black'],
      ['chrome-silver', 'vintage-red', 'grid-gray'],
      ['obsidian', 'plasma-orange', 'quantum-teal'],
      ['cosmic-purple', 'iridescent-pink', 'stardust-white']
    ];

    const heats = [89, 92, 85, 94, 87, 91];

    const newCard = {
      id,
      title: baseTitle,
      prompt: prompts[idx % 6],
      colorTags: colors[idx % 6],
      heat: heats[idx % 6]
    };

    cards[idx] = newCard;
  }

  function filteredCards() {
    if (filter === 'favorites') {
      return cards.filter(card => favorites.has(card.id));
    }
    return cards;
  }

  // Mood options
  const moods = [
    { id: 'creative', label: 'Creative', icon: '✨' },
    { id: 'calm', label: 'Calm', icon: '🧘' },
    { id: 'energetic', label: 'Energetic', icon: '⚡' },
    { id: 'mystical', label: 'Mystical', icon: '🔮' },
    { id: 'retro', label: 'Retro', icon: '📼' }
  ];
</script>

<style lang="scss">
  :global(body) {
    margin: 0;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #fff;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  .app-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  .header {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  .logo {
    width: 64px;
    height: 64px;
    margin: 0 auto 1rem;
    filter: drop-shadow(0 0 16px rgba(109, 40, 217, 0.5));
  }

  h1 {
    font-weight: 800;
    font-size: 2.5rem;
    margin: 0.5rem 0 0.75rem;
    background: linear-gradient(90deg, #a855f7, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    letter-spacing: -0.02em;
  }

  .subtitle {
    font-weight: 300;
    opacity: 0.85;
    font-size: 1.125rem;
    max-width: 600px;
    margin: 0 auto 2rem;
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 1.25rem;
    margin-bottom: 2.5rem;
    justify-content: center;
  }

  .input-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-width: 240px;
  }

  label {
    font-weight: 500;
    font-size: 0.875rem;
    opacity: 0.9;
  }

  input,
  select {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    color: white;
    font-size: 1rem;
    transition: all 0.3s ease;
  }

  input:focus,
  select:focus {
    outline: none;
    border-color: #8b5cf6;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
  }

  input::placeholder {
    color: rgba(255, 255, 255, 0.4);
  }

  .intensity-slider {
    width: 100%;
  }

  .slider-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    opacity: 0.7;
  }

  .btn {
    background: linear-gradient(90deg, #8b5cf6, #a855f7);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.3);
  }

  .btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.45);
  }

  .btn:active:not(:disabled) {
    transform: translateY(0);
  }

  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .filters {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    margin-bottom: 2rem;
  }

  .filter-btn {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #fff;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.25s ease;
  }

  .filter-btn.active {
    background: rgba(139, 92, 246, 0.25);
    border-color: #8b5cf6;
    color: #8b5cf6;
  }

  .cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1.5rem;
  }

  .card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
  }

  .card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 70%);
    z-index: -1;
  }

  .card:hover {
    transform: translateY(-4px);
    border-color: rgba(139, 92, 246, 0.3);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
  }

  .card-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
  }

  .card-actions {
    display: flex;
    gap: 0.5rem;
  }

  .action-btn {
    background: rgba(255, 255, 255, 0.06);
    border: none;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-btn:hover:not(:disabled) {
    background: rgba(139, 92, 246, 0.25);
    transform: scale(1.1);
  }

  .action-btn:disabled {
    opacity: 0.4;
  }

  .card-prompt {
    font-size: 0.95rem;
    opacity: 0.85;
    margin-bottom: 1.25rem;
    line-height: 1.5;
  }

  .color-tags {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .color-tag {
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .neon-cyan { background: rgba(0, 220, 255, 0.15); color: #00dcff; border: 1px solid rgba(0, 220, 255, 0.3); }
  .midnight-blue { background: rgba(25, 25, 112, 0.25); color: #191970; border: 1px solid rgba(25, 25, 112, 0.4); }
  .electric-purple { background: rgba(176, 106, 252, 0.15); color: #b06afc; border: 1px solid rgba(176, 106, 252, 0.3); }
  .slate-gray { background: rgba(112, 128, 144, 0.15); color: #708090; border: 1px solid rgba(112, 128, 144, 0.3); }
  .amber-glow { background: rgba(255, 191, 0, 0.2); color: #ffbfb0; border: 1px solid rgba(255, 191, 0, 0.3); }
  .smoke-white { background: rgba(245, 245, 245, 0.1); color: #f5f5f5; border: 1px solid rgba(245, 245, 245, 0.2); }
  .deep-indigo { background: rgba(75, 0, 130, 0.25); color: #4b0082; border: 1px solid rgba(75, 0, 130, 0.4); }
  .bioluminescent-green { background: rgba(0, 255, 127, 0.15); color: #00ff7f; border: 1px solid rgba(0, 255, 127, 0.3); }
  .void-black { background: rgba(0, 0, 0, 0.3); color: #000; border: 1px solid rgba(0, 0, 0, 0.4); }
  .chrome-silver { background: rgba(192, 192, 192, 0.15); color: #c0c0c0; border: 1px solid rgba(192, 192, 192, 0.3); }
  .vintage-red { background: rgba(220, 20, 60, 0.15); color: #dc143c; border: 1px solid rgba(220, 20, 60, 0.3); }
  .grid-gray { background: rgba(128, 128, 128, 0.15); color: #808080; border: 1px solid rgba(128, 128, 128, 0.3); }
  .obsidian { background: rgba(30, 30, 30, 0.3); color: #1e1e1e; border: 1px solid rgba(30, 30, 30, 0.4); }
  .plasma-orange { background: rgba(255, 105, 180, 0.15); color: #ff69b4; border: 1px solid rgba(255, 105, 180, 0.3); }
  .quantum-teal { background: rgba(0, 207, 192, 0.15); color: #00cfc0; border: 1px solid rgba(0, 207, 192, 0.3); }
  .cosmic-purple { background: rgba(147, 112, 219, 0.15); color: #9370db; border: 1px solid rgba(147, 112, 219, 0.3); }
  .iridescent-pink { background: rgba(255, 105, 180, 0.15); color: #ff69b4; border: 1px solid rgba(255, 105, 180, 0.3); }
  .stardust-white { background: rgba(255, 255, 255, 0.1); color: #fff; border: 1px solid rgba(255, 255, 255, 0.2); }

  .heat-meter {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .heat-bar {
    flex: 1;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
  }

  .heat-fill {
    height: 100%;
    background: linear-gradient(90deg, #ec4899, #8b5cf6);
    border-radius: 999px;
    transition: width 0.5s ease;
  }

  .heat-value {
    font-size: 0.875rem;
    font-weight: 600;
    min-width: 40px;
    text-align: right;
  }

  .empty-state {
    text-align: center;
    padding: 3rem 1rem;
    grid-column: 1 / -1;
  }

  .empty-state p {
    opacity: 0.7;
    margin-top: 1rem;
  }

  @media (max-width: 768px) {
    .controls {
      flex-direction: column;
      align-items: stretch;
    }

    .input-group {
      min-width: auto;
    }

    .cards-grid {
      grid-template-columns: 1fr;
    }
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .card {
    animation: fadeInUp 0.6s ease forwards;
  }

  .card:nth-child(1) { animation-delay: 0.1s; }
  .card:nth-child(2) { animation-delay: 0.2s; }
  .card:nth-child(3) { animation-delay: 0.3s; }
  .card:nth-child(4) { animation-delay: 0.4s; }
  .card:nth-child(5) { animation-delay: 0.5s; }
  .card:nth-child(6) { animation-delay: 0.6s; }

  .generating-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    backdrop-filter: blur(4px);
  }

  .spinner {
    width: 60px;
    height: 60px;
    border: 6px solid rgba(255, 255, 255, 0.1);
    border-top-color: #8b5cf6;
    border-radius: 50%;
    animation: spin 1s ease-in-out infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .spinner-text {
    margin-top: 1.5rem;
    font-size: 1.1rem;
    font-weight: 500;
  }
</style>

<div class="app-container">
  <header class="header">
    <div class="logo">🧠</div>
    <h1>AI Moodboard Studio</h1>
    <p class="subtitle">Generate stunning visual inspiration powered by AI — instantly, beautifully, endlessly.</p>
  </header>

  <section class="controls">
    <div class="input-group">
      <label for="theme">Theme</label>
      <input
        id="theme"
        type="text"
        bind:value={theme}
        placeholder="e.g., cyberpunk, nature, luxury..."
      />
    </div>

    <div class="input-group">
      <label for="mood">Mood Style</label>
      <select id="mood" bind:value={mood}>
        {#each moods as moodOpt}
          <option value={moodOpt.id}>{moodOpt.icon} {moodOpt.label}</option>
        {/each}
      </select>
    </div>

    <div class="input-group">
      <label for="intensity">Color Intensity</label>
      <input
        id="intensity"
        type="range"
        class="intensity-slider"
        bind:value={intensity}
        min="30"
        max="100"
      />
      <div class="slider-label">
        <span>Subtle</span>
        <span>Intense</span>
      </div>
    </div>

    <div class="input-group">
      <label>&nbsp;</label>
      <button class="btn" on:click={generateCards} disabled={isGenerating || !theme.trim()}>
        {#if isGenerating}
          <span class="spinner"></span>
          Generating...
        {:else}
          🌟 Generate Moodboard
        {/if}
      </button>
    </div>
  </section>

  <section class="filters">
    <button
      class="filter-btn {filter === 'all' ? 'active' : ''}"
      on:click={() => filter = 'all'}
    >
      All ({cards.length})
    </button>
    <button
      class="filter-btn {filter === 'favorites' ? 'active' : ''}"
      on:click={() => filter = 'favorites'}
    >
      Favorites ({Array.from(favorites).length})
    </button>
  </section>

  {#if cards.length === 0}
    <div class="empty-state">
      <h2>✨ Your Moodboard Awaits</h2>
      <p>Enter a theme above and click "Generate Moodboard" to create your first set of AI-powered inspiration cards.</p>
    </div>
  {:else}
    <section class="cards-grid">
      {#each filteredCards() as card (card.id)}
        <article class="card">
          <div class="card-header">
            <h3 class="card-title">{card.title}</h3>
            <div class="card-actions">
              <button
                class="action-btn"
                on:click={() => toggleFavorite(card.id)}
                aria-label={favorites.has(card.id) ? 'Remove from favorites' : 'Add to favorites'}
              >
                {#if favorites.has(card.id)}
                  ⭐
                {:else}
                  ☆
                {/if}
              </button>
              <button
                class="action-btn"
                on:click={() => refreshCard(card.id)}
                aria-label="Refresh this card"
              >
                ↻
              </button>
            </div>
          </div>

          <p class="card-prompt">{card.prompt}</p>

          <div class="color-tags">
            {#each card.colorTags as tag}
              <span class="color-tag {tag}">{tag.replace('-', ' ')}</span>
            {/each}
          </div>

          <div class="heat-meter">
            <span class="heat-value">{card.heat}%</span>
            <div class="heat-bar">
              <div class="heat-fill" style="width: {card.heat}%"></div>
            </div>
          </div>
        </article>
      {/each}
    </section>
  {/if}
</div>

{#if isGenerating}
  <div class="generating-overlay">
    <div>
      <div class="spinner"></div>
      <p class="spinner-text">AI is crafting your moodboard...</p>
    </div>
  </div>
{/if}