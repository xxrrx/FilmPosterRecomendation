import MovieCard from './MovieCard'

function SkeletonCard() {
  return (
    <div className="bg-[#16213e] rounded-lg overflow-hidden">
      <div className="skeleton aspect-[2/3]" />
      <div className="p-3 space-y-2">
        <div className="skeleton h-4 rounded w-3/4" />
        <div className="skeleton h-3 rounded w-1/4" />
      </div>
    </div>
  )
}

export default function MovieGrid({ movies, loading, skeletonCount = 20 }) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {Array.from({ length: skeletonCount }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    )
  }

  if (!movies?.length) {
    return (
      <div className="text-center text-gray-400 py-20 text-lg">
        Khong tim thay phim nao.
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
      {movies.map(m => (
        <MovieCard key={m.movie_id} movie={m} />
      ))}
    </div>
  )
}
