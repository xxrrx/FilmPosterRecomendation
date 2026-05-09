import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { getMovies } from '../api/movieApi'
import MovieGrid from '../components/MovieGrid'
import SearchBar from '../components/SearchBar'
import GenreTag from '../components/GenreTag'

const GENRES = [
  'Action','Adventure','Animation','Comedy','Crime',
  'Drama','Family','Fantasy','Horror','Mystery',
  'Romance','Science Fiction','Thriller','War',
]

export default function HomePage() {
  const [movies, setMovies]       = useState([])
  const [loading, setLoading]     = useState(true)
  const [page, setPage]           = useState(1)
  const [total, setTotal]         = useState(0)
  const [activeGenre, setGenre]   = useState(null)
  const LIMIT = 20

  const load = useCallback(async (p, genre) => {
    setLoading(true)
    try {
      const data = await getMovies(p, LIMIT, genre)
      setMovies(data.movies || [])
      setTotal(data.total || 0)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load(page, activeGenre) }, [page, activeGenre, load])

  const handleGenre = (g) => {
    const next = activeGenre === g ? null : g
    setGenre(next); setPage(1)
  }

  const totalPages = Math.ceil(total / LIMIT)

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <div className="bg-gradient-to-b from-[#0f3460] to-[#1a1a2e] py-12 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-4xl font-bold text-white mb-2">
            🎬 Movie <span className="text-[#e94560]">Recommender</span>
          </h1>
          <p className="text-gray-300 mb-8 text-sm">
            Goi y phim thong minh: CNN poster + TF-IDF + K-Means + Naive Bayes
          </p>
          <div className="flex justify-center">
            <SearchBar />
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Genre filter */}
        <div className="mb-6">
          <h2 className="text-gray-300 text-sm font-semibold mb-3">Loc theo the loai:</h2>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => handleGenre(null)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors
                ${!activeGenre ? 'bg-[#e94560] text-white' : 'bg-[#16213e] text-gray-300 hover:bg-[#0f3460]'}`}
            >
              Tat ca
            </button>
            {GENRES.map(g => (
              <GenreTag key={g} genre={g} active={activeGenre === g} onClick={handleGenre} />
            ))}
          </div>
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between mb-4">
          <p className="text-gray-400 text-sm">
            {activeGenre ? `The loai: ${activeGenre} — ` : ''}
            {total} phim
          </p>
          <Link to="/explore" className="text-[#e94560] text-sm hover:underline">
            Kham pha cum phim →
          </Link>
        </div>

        {/* Grid */}
        <MovieGrid movies={movies} loading={loading} />

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex justify-center items-center gap-2 mt-8">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-4 py-2 bg-[#16213e] text-white rounded-lg disabled:opacity-40 hover:bg-[#0f3460] transition-colors text-sm"
            >
              ← Truoc
            </button>
            <span className="text-gray-400 text-sm px-4">
              Trang {page} / {totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-4 py-2 bg-[#16213e] text-white rounded-lg disabled:opacity-40 hover:bg-[#0f3460] transition-colors text-sm"
            >
              Sau →
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
