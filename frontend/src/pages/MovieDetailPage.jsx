import { useState, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { getMovie, getRecommendations } from '../api/movieApi'
import GenreTag from '../components/GenreTag'
import WeightSlider from '../components/WeightSlider'
import MovieCard from '../components/MovieCard'

const PLACEHOLDER = 'https://via.placeholder.com/400x600/0f3460/e0e0e0?text=No+Poster'

export default function MovieDetailPage() {
  const { id } = useParams()
  const navigate = useNavigate()

  const [movie, setMovie]       = useState(null)
  const [recs, setRecs]         = useState([])
  const [alpha, setAlpha]       = useState(0.5)
  const [loadingMovie, setLM]   = useState(true)
  const [loadingRecs, setLR]    = useState(true)
  const [error, setError]       = useState(null)

  // Load movie detail
  useEffect(() => {
    setLM(true); setError(null)
    getMovie(id)
      .then(setMovie)
      .catch(() => setError('Khong tim thay phim.'))
      .finally(() => setLM(false))
  }, [id])

  // Load recommendations khi alpha thay doi
  useEffect(() => {
    if (!id) return
    setLR(true)
    getRecommendations(id, alpha, 10)
      .then(d => setRecs(d.recommendations || []))
      .catch(() => setRecs([]))
      .finally(() => setLR(false))
  }, [id, alpha])

  if (loadingMovie) return (
    <div className="flex justify-center items-center min-h-screen">
      <div className="w-10 h-10 border-4 border-[#e94560] border-t-transparent rounded-full animate-spin" />
    </div>
  )

  if (error) return (
    <div className="text-center py-20 text-gray-400">
      <p className="text-xl mb-4">{error}</p>
      <button onClick={() => navigate(-1)} className="text-[#e94560] hover:underline">← Quay lai</button>
    </div>
  )

  if (!movie) return null

  let genres = []
  try { genres = JSON.parse(movie.genres) } catch {}

  let predictedGenres = []
  try { predictedGenres = JSON.parse(movie.predicted_genres || '[]') } catch {}

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Back */}
      <button onClick={() => navigate(-1)} className="text-gray-400 hover:text-white mb-6 flex items-center gap-2 text-sm transition-colors">
        ← Quay lai
      </button>

      {/* Detail section */}
      <div className="flex flex-col md:flex-row gap-8 mb-10">
        {/* Poster */}
        <div className="shrink-0">
          <img
            src={movie.poster_url || PLACEHOLDER}
            alt={movie.title}
            className="w-56 rounded-xl shadow-2xl mx-auto md:mx-0"
            onError={e => { e.target.src = PLACEHOLDER }}
          />
        </div>

        {/* Info */}
        <div className="flex-1 space-y-4">
          <div>
            <h1 className="text-3xl font-bold text-white mb-1">{movie.title}</h1>
            <p className="text-gray-400 text-sm">{movie.year} · ★ {movie.rating?.toFixed(1)} · {movie.vote_count?.toLocaleString()} luot vote</p>
          </div>

          {/* Actual genres */}
          <div>
            <p className="text-xs text-gray-500 mb-1 uppercase tracking-wide">The loai</p>
            <div className="flex flex-wrap gap-2">
              {genres.map(g => <GenreTag key={g} genre={g} />)}
            </div>
          </div>

          {/* NB predicted genres */}
          {predictedGenres.length > 0 && (
            <div className="bg-[#16213e] border border-[#e94560]/30 rounded-xl p-3">
              <p className="text-xs text-[#e94560] font-semibold mb-2">🤖 The loai du doan boi AI (Naive Bayes)</p>
              <div className="flex flex-wrap gap-2">
                {predictedGenres.map(g => (
                  <span key={g} className="px-2 py-0.5 bg-[#e94560]/20 text-[#e94560] text-xs rounded border border-[#e94560]/40">
                    {g}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Cluster info */}
          <div className="flex items-center gap-3">
            <div className="bg-[#0f3460] rounded-lg px-3 py-2 text-sm">
              <span className="text-gray-400">Cum K-Means: </span>
              <Link
                to={`/explore?cluster=${movie.cluster_id}`}
                className="text-[#e94560] font-bold hover:underline"
              >
                #{movie.cluster_id}
              </Link>
            </div>
          </div>

          {/* Overview */}
          {movie.overview && (
            <div>
              <p className="text-xs text-gray-500 mb-1 uppercase tracking-wide">Mo ta</p>
              <p className="text-gray-300 text-sm leading-relaxed line-clamp-6">
                {movie.overview}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Weight slider */}
      <div className="mb-8 max-w-md">
        <h2 className="text-white font-semibold mb-3">Dieu chinh trong so goi y</h2>
        <WeightSlider alpha={alpha} onChange={setAlpha} />
      </div>

      {/* Recommendations */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-white font-semibold text-lg">
            Top 10 phim goi y
            {loadingRecs && (
              <span className="ml-2 w-4 h-4 border-2 border-[#e94560] border-t-transparent rounded-full animate-spin inline-block align-middle" />
            )}
          </h2>
          <span className="text-gray-400 text-xs">alpha={alpha.toFixed(2)}</span>
        </div>

        {recs.length === 0 && !loadingRecs ? (
          <p className="text-gray-400 text-sm">Khong co goi y.</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {recs.map(m => (
              <div key={m.movie_id} className="relative">
                <MovieCard movie={m} />
                <div className="absolute bottom-[52px] right-2 bg-black/70 text-xs text-green-400 px-1.5 py-0.5 rounded font-mono">
                  {(m.similarity * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
