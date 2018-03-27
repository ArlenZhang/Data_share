def random_normal_advanced(loc, scale, size, edus_num, tmp_idx):
    numpy_rand_normal = list(np.random.normal(loc, scale, size))
    print(numpy_rand_normal)
    max_ = max(numpy_rand_normal)
    min_ = min(numpy_rand_normal)
    print(max_)
    print(min_)
    # 进行实际转换
    piece_ = edus_num/(max_-min_)
    center_ = (max_ + min_)/2
    result_normal_sample = []
    for rand in numpy_rand_normal:
        val_ = int(piece_*(rand-center_)+tmp_idx)
        if val_ >= 0 or val_ < edus_num:  # 对于合法样本的保留，允许将不合法的edu进行丢弃
            result_normal_sample.append(val_)
    return result_normal_sample
