export default function WeightSlider({ alpha, onChange }) {
  return (
    <div className="bg-[#16213e] rounded-xl p-4 border border-gray-700">
      <div className="flex justify-between items-center mb-3">
        <span className="text-sm font-semibold text-white">Trong so goi y</span>
        <span className="text-xs text-gray-400">alpha = {alpha.toFixed(2)}</span>
      </div>

      <input
        type="range" min="0" max="1" step="0.05"
        value={alpha}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full accent-[#e94560] cursor-pointer"
      />

      <div className="flex justify-between text-xs mt-2">
        <span className="text-blue-400">
          🖼 Poster ({((1 - alpha) * 100).toFixed(0)}% text)
        </span>
        <span className="text-pink-400">
          ({(alpha * 100).toFixed(0)}% poster) Poster 🖼
        </span>
      </div>

      <div className="mt-2 h-2 rounded-full bg-gradient-to-r from-blue-500 to-pink-500 relative">
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow"
          style={{ left: `calc(${alpha * 100}% - 6px)` }}
        />
      </div>

      <p className="text-xs text-gray-500 mt-2 text-center">
        {alpha < 0.3 ? 'Uu tien noi dung van ban' :
         alpha > 0.7 ? 'Uu tien hinh anh poster' :
         'Can bang poster & van ban'}
      </p>
    </div>
  )
}
