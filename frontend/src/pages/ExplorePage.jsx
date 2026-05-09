import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getClusters, getClusterMovies, getRules } from '../api/movieApi'
import ClusterChart from '../components/ClusterChart'
import RulesTable from '../components/RulesTable'
import MovieGrid from '../components/MovieGrid'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts'

const COLORS = [
  '#e94560','#00b4d8','#06d6a0','#ffd166','#ef476f',
  '#118ab2','#f77f00','#a8dadc','#457b9d','#e63946',
  '#2a9d8f','#e9c46a','#f4a261','#264653','#b5e48c',
  '#99d98c','#76c893','#52b69a','#34a0a4','#168aad',
]

export default function ExplorePage() {
  const [searchParams] = useSearchParams()
  const initCluster = searchParams.get('cluster') ? parseInt(searchParams.get('cluster')) : null

  const [clusters, setClusters]           = useState([])
  const [selectedCluster, setSelected]    = useState(initCluster)
  const [clusterMovies, setClusterMovies] = useState([])
  const [allMovies, setAllMovies]         = useState([])
  const [rules, setRules]                 = useState([])
  const [tab, setTab]                     = useState('chart')
  const [loading, setLoading]             = useState(true)
  const [loadingMovies, setLM]            = useState(false)

  // Load clusters & rules
  useEffect(() => {
    Promise.all([getClusters(), getRules(0, 1.0, 100)])
      .then(([cData, rData]) => {
        setClusters(cData.clusters || [])
        setRules(rData.rules || [])
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  // Load all movies for scatter plot (lay tung cluster)
  useEffect(() => {
    if (!clusters.length) return
    Promise.all(clusters.map(c => getClusterMovies(c.cluster_id, 200)))
      .then(results => {
        const all = results.flatMap(r => r.movies || [])
        setAllMovies(all)
      })
      .catch(console.error)
  }, [clusters])

  // Load movies khi chon cluster
  useEffect(() => {
    if (selectedCluster === null) { setClusterMovies([]); return }
    setLM(true)
    getClusterMovies(selectedCluster, 50)
      .then(d => setClusterMovies(d.movies || []))
      .catch(() => setClusterMovies([]))
      .finally(() => setLM(false))
  }, [selectedCluster])

  const barData = clusters.map(c => ({
    name: `C${c.cluster_id}`,
    count: c.movie_count,
    cluster_id: c.cluster_id,
    top: c.top_genres?.join(', ') || '',
  }))

  const CustomBarTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
      <div className="bg-[#0f3460] border border-gray-700 rounded p-2 text-xs">
        <p className="text-white font-bold">{d.name}</p>
        <p className="text-gray-300">{d.count} phim</p>
        <p className="text-[#e94560]">{d.top}</p>
      </div>
    )
  }

  if (loading) return (
    <div className="flex justify-center items-center min-h-screen">
      <div className="w-10 h-10 border-4 border-[#e94560] border-t-transparent rounded-full animate-spin" />
    </div>
  )

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-white mb-2">Kham Pha Phim</h1>
      <p className="text-gray-400 text-sm mb-6">
        Truc quan hoa {clusters.length} cum K-Means · {rules.length} luat ket hop
      </p>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-700 pb-2">
        {[
          { key: 'chart', label: '🗺 PCA 2D' },
          { key: 'bar', label: '📊 So luong cum' },
          { key: 'rules', label: '🔗 Association Rules' },
        ].map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 rounded-t text-sm font-medium transition-colors
              ${tab === t.key ? 'bg-[#e94560] text-white' : 'text-gray-400 hover:text-white'}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* PCA Scatter */}
      {tab === 'chart' && (
        <div>
          {allMovies.length > 0
            ? <ClusterChart movies={allMovies} />
            : <div className="text-center text-gray-400 py-20">Dang tai du lieu...</div>
          }
        </div>
      )}

      {/* Bar chart */}
      {tab === 'bar' && (
        <div className="bg-[#16213e] rounded-xl p-4 border border-gray-700">
          <h3 className="text-white font-semibold mb-4">So phim trong moi cum</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={barData} margin={{ top: 5, right: 10, left: 0, bottom: 20 }}>
              <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <Tooltip content={<CustomBarTooltip />} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]} onClick={d => setSelected(d.cluster_id)}>
                {barData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={COLORS[entry.cluster_id % COLORS.length]}
                    opacity={selectedCluster === null || selectedCluster === entry.cluster_id ? 1 : 0.4}
                    style={{ cursor: 'pointer' }}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 text-center mt-2">Click vao cot de xem phim trong cum</p>
        </div>
      )}

      {/* Association Rules */}
      {tab === 'rules' && (
        <div>
          <div className="bg-[#16213e] rounded-xl p-4 border border-gray-700 mb-4">
            <p className="text-gray-300 text-sm">
              <span className="text-[#e94560] font-semibold">Luat ket hop the loai</span> —
              Khi nguoi xem thich the loai A, ho thuong cung thich the loai B.
            </p>
          </div>
          <RulesTable rules={rules} />
        </div>
      )}

      {/* Cluster movies */}
      {selectedCluster !== null && tab === 'bar' && (
        <div className="mt-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-white font-semibold">
              Phim trong Cum #{selectedCluster}
              {clusters.find(c => c.cluster_id === selectedCluster)?.top_genres && (
                <span className="text-gray-400 text-sm ml-2">
                  ({clusters.find(c => c.cluster_id === selectedCluster)?.top_genres.join(', ')})
                </span>
              )}
            </h2>
            <button onClick={() => setSelected(null)} className="text-gray-400 hover:text-white text-sm">
              Dong ×
            </button>
          </div>
          <MovieGrid movies={clusterMovies} loading={loadingMovies} skeletonCount={10} />
        </div>
      )}
    </div>
  )
}
