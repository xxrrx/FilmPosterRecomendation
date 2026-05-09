import { useState, useMemo } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { useNavigate } from 'react-router-dom'

const COLORS = [
  '#e94560','#00b4d8','#06d6a0','#ffd166','#ef476f',
  '#118ab2','#073b4c','#f77f00','#a8dadc','#457b9d',
  '#1d3557','#e63946','#2a9d8f','#e9c46a','#f4a261',
  '#264653','#b5e48c','#99d98c','#76c893','#52b69a',
]

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-[#0f3460] border border-gray-600 rounded-lg p-3 text-xs shadow-xl max-w-[200px]">
      <p className="text-white font-semibold truncate">{d.title}</p>
      <p className="text-gray-400">{d.year}</p>
      <p className="text-[#e94560]">Cum {d.cluster_id}</p>
      <p className="text-yellow-400">★ {d.rating?.toFixed(1)}</p>
    </div>
  )
}

export default function ClusterChart({ movies }) {
  const navigate = useNavigate()
  const [hoveredCluster, setHoveredCluster] = useState(null)

  const data = useMemo(() =>
    movies.map(m => ({
      x: m.pca_x,
      y: m.pca_y,
      cluster_id: m.cluster_id,
      title: m.title,
      year: m.year,
      rating: m.rating,
      movie_id: m.movie_id,
    })),
    [movies]
  )

  return (
    <div className="bg-[#16213e] rounded-xl p-4 border border-gray-700">
      <h3 className="text-white font-semibold mb-1">PCA 2D — Phan bo cum phim</h3>
      <p className="text-gray-400 text-xs mb-4">Moi diem = 1 phim. Mau = cum K-Means. Click de xem chi tiet.</p>

      <ResponsiveContainer width="100%" height={420}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <XAxis dataKey="x" type="number" hide />
          <YAxis dataKey="y" type="number" hide />
          <Tooltip content={<CustomTooltip />} />
          <Scatter
            data={data}
            onClick={d => navigate(`/movie/${d.movie_id}`)}
          >
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={COLORS[entry.cluster_id % COLORS.length]}
                opacity={hoveredCluster === null || hoveredCluster === entry.cluster_id ? 0.8 : 0.15}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredCluster(entry.cluster_id)}
                onMouseLeave={() => setHoveredCluster(null)}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
