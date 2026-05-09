const GENRE_COLORS = {
  Action: 'bg-red-700', Adventure: 'bg-orange-600', Animation: 'bg-yellow-600',
  Comedy: 'bg-green-600', Crime: 'bg-gray-600', Documentary: 'bg-teal-600',
  Drama: 'bg-blue-700', Family: 'bg-pink-600', Fantasy: 'bg-purple-700',
  History: 'bg-amber-700', Horror: 'bg-red-900', Music: 'bg-indigo-600',
  Mystery: 'bg-slate-600', Romance: 'bg-rose-600', 'Science Fiction': 'bg-cyan-700',
  Thriller: 'bg-zinc-700', War: 'bg-stone-600', Western: 'bg-yellow-800',
}

export default function GenreTag({ genre, onClick, active = false }) {
  const color = GENRE_COLORS[genre] || 'bg-gray-700'
  return (
    <span
      onClick={() => onClick?.(genre)}
      className={`inline-block px-2 py-0.5 rounded text-xs font-medium text-white cursor-pointer
        ${color} ${active ? 'ring-2 ring-white' : 'opacity-80 hover:opacity-100'}
        transition-opacity`}
    >
      {genre}
    </span>
  )
}
