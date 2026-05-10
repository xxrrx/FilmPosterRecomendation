# Tài Liệu Tham Khảo

## Sách và Bài Báo Khoa Học

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778, 2016.

[2] C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008. Chương 6: Scoring, term weighting and the vector space model.

[3] S. J. Pan and Q. Yang, "A Survey on Transfer Learning," *IEEE Transactions on Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359, Oct. 2010.

[4] R. Agrawal and R. Srikant, "Fast Algorithms for Mining Association Rules," in *Proceedings of the 20th International Conference on Very Large Data Bases (VLDB)*, pp. 487–499, 1994.

[5] J. A. Hartigan and M. A. Wong, "Algorithm AS 136: A K-Means Clustering Algorithm," *Journal of the Royal Statistical Society. Series C (Applied Statistics)*, vol. 28, no. 1, pp. 100–108, 1979.

[6] G. H. John and P. Langley, "Estimating Continuous Distributions in Bayesian Classifiers," in *Proceedings of the Eleventh Conference on Uncertainty in Artificial Intelligence (UAI)*, pp. 338–345, 1995.

[7] T. Zhang, R. Ramakrishnan, and M. Livny, "BIRCH: An Efficient Data Clustering Method for Very Large Databases," *ACM SIGMOD Record*, vol. 25, no. 2, pp. 103–114, 1996.

[8] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[9] M. Abadi et al., "TensorFlow: A System for Large-Scale Machine Learning," in *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, pp. 265–283, 2016.

[10] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 3982–3992, 2019.

## Hệ Thống Gợi Ý

[11] Y. Koren, R. Bell, and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," *IEEE Computer*, vol. 42, no. 8, pp. 30–37, Aug. 2009.

[12] G. Adomavicius and A. Tuzhilin, "Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions," *IEEE Transactions on Knowledge and Data Engineering*, vol. 17, no. 6, pp. 734–749, Jun. 2005.

[13] X. Su and T. M. Khoshgoftaar, "A Survey of Collaborative Filtering Techniques," *Advances in Artificial Intelligence*, vol. 2009, Article ID 421425, 19 pages, 2009.

[14] M. J. Pazzani and D. Billsus, "Content-Based Recommendation Systems," in *The Adaptive Web: Methods and Strategies of Web Personalization*, Springer-Verlag, pp. 325–341, 2007.

## Tập Dữ Liệu và Công Cụ

[15] The Movie Database (TMDB), "TMDB 5000 Movie Dataset," Kaggle, 2017. [Online]. Available: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

[16] S. Bird, E. Loper, and E. Klein, *Natural Language Processing with Python*, O'Reilly Media Inc., 2009.

[17] S. Raschka, "MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack," *Journal of Open Source Software*, vol. 3, no. 24, p. 638, 2018.

[18] W. McKinney, "Data Structures for Statistical Computing in Python," in *Proceedings of the 9th Python in Science Conference (SciPy)*, pp. 56–61, 2010.

## Deep Learning và Computer Vision

[19] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 25, pp. 1097–1105, 2012.

[20] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2015.

[21] M. Lin, Q. Chen, and S. Yan, "Network In Network," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2014. (Giới thiệu Global Average Pooling)

[22] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021.

## Backend và Frontend

[23] S. Ramírez, "FastAPI," GitHub, 2018. [Online]. Available: https://github.com/tiangolo/fastapi

[24] Meta Platforms, "React: A JavaScript library for building user interfaces," GitHub, 2013. [Online]. Available: https://github.com/facebook/react

[25] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, Jul. 2021. (FAISS)

---

## Ghi Chú

Tất cả số liệu và kết quả trong báo cáo này được lấy trực tiếp từ các thực nghiệm thực tế trên tập dữ liệu TMDB 5000 Movies, thực hiện trong các notebook `01_data_collection.ipynb`, `02_feature_extraction.ipynb`, và `03_ml_models.ipynb`.

Mã nguồn đầy đủ của dự án KhaiPha có tại: `C:\Users\ACER\OneDrive\Desktop\KhaiPha`

Báo cáo được viết bằng tiếng Việt và có thể chuyển đổi sang PDF hoặc DOCX bằng Pandoc:

```bash
# Chuyển đổi sang PDF (cần LaTeX)
pandoc report/*.md -o khaipah_report.pdf --pdf-engine=xelatex

# Chuyển đổi sang DOCX
pandoc report/*.md -o khaipah_report.docx
```
