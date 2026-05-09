import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const getMovies = (page = 1, limit = 20, genre = null) => {
  const params = { page, limit }
  if (genre) params.genre = genre
  return api.get('/movies', { params }).then(r => r.data)
}

export const searchMovies = (q, limit = 20) =>
  api.get('/movies/search', { params: { q, limit } }).then(r => r.data)

export const getMovie = (id) =>
  api.get(`/movies/${id}`).then(r => r.data)

export const getRecommendations = (id, alpha = 0.5, top_k = 10) =>
  api.get(`/movies/${id}/recommend`, { params: { alpha, top_k } }).then(r => r.data)

export const getRules = (min_confidence = 0, min_lift = 1.0, limit = 50) =>
  api.get('/genres/rules', { params: { min_confidence, min_lift, limit } }).then(r => r.data)

export const getClusters = () =>
  api.get('/clusters').then(r => r.data)

export const getClusterMovies = (clusterId, limit = 50) =>
  api.get(`/clusters/${clusterId}/movies`, { params: { limit } }).then(r => r.data)
