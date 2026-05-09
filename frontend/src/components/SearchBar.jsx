import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { searchMovies } from '../api/movieApi'

export default function SearchBar() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const ref = useRef()

  // Debounce search
  useEffect(() => {
    if (!query.trim()) { setResults([]); setOpen(false); return }
    const t = setTimeout(async () => {
      setLoading(true)
      try {
        const data = await searchMovies(query, 8)
        setResults(data.movies || [])
        setOpen(true)
      } catch { setResults([]) }
      finally { setLoading(false) }
    }, 350)
    return () => clearTimeout(t)
  }, [query])

  // Close on outside click
  useEffect(() => {
    const handler = e => { if (!ref.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const go = (movie) => {
    setQuery(''); setOpen(false)
    navigate(`/movie/${movie.movie_id}`)
  }

  return (
    <div ref={ref} className="relative w-full max-w-xl">
      <div className="flex items-center bg-[#16213e] border border-gray-600 rounded-full px-4 py-2">
        <svg className="w-4 h-4 text-gray-400 mr-2 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
        </svg>
        <input
          className="bg-transparent outline-none text-white placeholder-gray-400 text-sm w-full"
          placeholder="Tim phim..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          onFocus={() => results.length && setOpen(true)}
        />
        {loading && <div className="w-4 h-4 border-2 border-[#e94560] border-t-transparent rounded-full animate-spin shrink-0" />}
      </div>

      {open && results.length > 0 && (
        <div className="absolute top-full mt-2 w-full bg-[#16213e] border border-gray-700 rounded-lg shadow-xl z-50 overflow-hidden">
          {results.map(m => {
            let genres = []
            try { genres = JSON.parse(m.genres) } catch {}
            return (
              <div
                key={m.movie_id}
                onClick={() => go(m)}
                className="flex items-center gap-3 px-4 py-3 hover:bg-[#0f3460] cursor-pointer transition-colors"
              >
                <img
                  src={m.poster_url}
                  alt={m.title}
                  className="w-9 h-14 object-cover rounded shrink-0"
                  onError={e => e.target.style.display = 'none'}
                />
                <div className="min-w-0">
                  <p className="text-white text-sm font-medium truncate">{m.title}</p>
                  <p className="text-gray-400 text-xs">{m.year} · {genres.slice(0, 2).join(', ')}</p>
                </div>
                <span className="ml-auto text-yellow-400 text-xs shrink-0">★ {m.rating?.toFixed(1)}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
