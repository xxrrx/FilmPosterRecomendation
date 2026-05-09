import { useNavigate } from 'react-router-dom'
import GenreTag from './GenreTag'

const PLACEHOLDER = 'https://via.placeholder.com/200x300/0f3460/e0e0e0?text=No+Poster'

export default function MovieCard({ movie }) {
  const navigate = useNavigate()
  let genres = []
  try { genres = JSON.parse(movie.genres) } catch {}

  return (
    <div
      className="poster-card bg-[#16213e] rounded-lg overflow-hidden cursor-pointer"
      onClick={() => navigate(`/movie/${movie.movie_id}`)}
    >
      <div className="relative aspect-[2/3] overflow-hidden">
        <img
          src={movie.poster_url || PLACEHOLDER}
          alt={movie.title}
          className="w-full h-full object-cover"
          onError={e => { e.target.src = PLACEHOLDER }}
          loading="lazy"
        />
        {/* Rating badge */}
        <div className="absolute top-2 right-2 bg-black/70 text-yellow-400 text-xs font-bold px-1.5 py-0.5 rounded">
          ★ {movie.rating?.toFixed(1)}
        </div>
        {/* Cluster badge */}
        {movie.cluster_id >= 0 && (
          <div className="absolute top-2 left-2 bg-[#e94560]/80 text-white text-xs px-1.5 py-0.5 rounded">
            C{movie.cluster_id}
          </div>
        )}
      </div>

      <div className="p-3">
        <h3 className="text-sm font-semibold text-white truncate" title={movie.title}>
          {movie.title}
        </h3>
        <p className="text-xs text-gray-400 mb-2">{movie.year || '—'}</p>
        <div className="flex flex-wrap gap-1">
          {genres.slice(0, 2).map(g => (
            <GenreTag key={g} genre={g} />
          ))}
        </div>
      </div>
    </div>
  )
}
