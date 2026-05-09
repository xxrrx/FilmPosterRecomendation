export default function RulesTable({ rules }) {
  if (!rules?.length) return (
    <div className="text-center text-gray-400 py-8">Chua co luat ket hop.</div>
  )

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-700">
      <table className="w-full text-sm">
        <thead className="bg-[#0f3460] text-gray-300">
          <tr>
            <th className="px-4 py-3 text-left">Neu (Antecedent)</th>
            <th className="px-4 py-3 text-left">Thi (Consequent)</th>
            <th className="px-4 py-3 text-right">Support</th>
            <th className="px-4 py-3 text-right">Confidence</th>
            <th className="px-4 py-3 text-right">Lift</th>
          </tr>
        </thead>
        <tbody>
          {rules.map((r, i) => (
            <tr
              key={r.id}
              className={`border-t border-gray-700 hover:bg-[#0f3460]/50 transition-colors
                ${i % 2 === 0 ? 'bg-[#16213e]' : 'bg-[#1a1a2e]'}`}
            >
              <td className="px-4 py-3 text-[#e94560] font-medium">{r.antecedent}</td>
              <td className="px-4 py-3 text-blue-300">{r.consequent}</td>
              <td className="px-4 py-3 text-right text-gray-300">{(r.support * 100).toFixed(1)}%</td>
              <td className="px-4 py-3 text-right">
                <span className={`font-semibold ${r.confidence >= 0.7 ? 'text-green-400' : r.confidence >= 0.5 ? 'text-yellow-400' : 'text-gray-300'}`}>
                  {(r.confidence * 100).toFixed(1)}%
                </span>
              </td>
              <td className="px-4 py-3 text-right">
                <span className={`font-semibold ${r.lift >= 2 ? 'text-green-400' : r.lift >= 1.5 ? 'text-yellow-400' : 'text-gray-300'}`}>
                  {r.lift.toFixed(2)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
