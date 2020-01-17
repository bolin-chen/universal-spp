function z = universal_spatial_spp_vec_eval(x, y)

  f_n = 7;
  m = 4;

  col_1 = im2col(x, [1, f_n], 'sliding');
  col_2 = im2col(x', [1, f_n], 'sliding');
  col = cat(2, col_1, col_2);
  neighbor = cat(1, col(1 : floor(1 * f_n / 2), :), col(floor(1 * f_n / 2) + 2 : 1 * f_n, :));
  target = col(floor(1 * f_n / 2) + 1, :);

  sol = lsqlin(neighbor', target);

  base_f = -ones(1, f_n);
  base_f(1 : floor(1 * f_n / 2)) = sol(1 : floor(1 * f_n / 2));
  base_f(floor(1 * f_n / 2) + 2 : 1 * f_n) = sol(floor(1 * f_n / 2) + 1 : 1 * f_n - 1);

  f_array = {
    padarray(base_f, [floor(f_n / 2), 0]),
    padarray(base_f', [0, floor(f_n / 2)]),
    conv2(base_f, base_f') / -1,
  };

  z = y;

  f_array_length = length(f_array);
  [f_n1, f_n2] = size(f_array{1});

  f = zeros(f_array_length, f_n1, f_n2);
  for i = 1 : f_array_length
    f(i, :, :) = f_array{i};
  end

  [n1, n2] = size(x);

  r_x = zeros(f_array_length, n1 + f_n1 - 1, n2 + f_n2 - 1);
  r_z = zeros(f_array_length, n1 + f_n1 - 1, n2 + f_n2 - 1);
  for i = 1 : f_array_length
    f_item = f_array{i};
    r_x(i, :, :) = conv2(x, f_item, 'full');
    r_z(i, :, :) = conv2(z, f_item, 'full');
  end

  p_r_x = zeros(f_array_length, f_n1, f_n2);
  p_r_z = zeros(f_array_length, f_n1, f_n2);
  p_r_t = zeros(f_array_length, f_n1, f_n2);

  for i = 1 : n1
    for j = 1 : n2

      delta = sign(x(i, j) - y(i, j)) * m;

      if (delta == 0) || (z(i, j) + delta < 0) || (z(i, j) + delta > 255)
        continue;
      end

      p_r_x = r_x(:, i : i + f_n1 - 1, j : j + f_n2 - 1);
      p_r_z = r_z(:, i : i + f_n1 - 1, j : j + f_n2 - 1);
      p_r_t = p_r_z + f * delta;

      p_d_tx = sum(sum(sum(abs(p_r_t - p_r_x))));
      p_d_zx = sum(sum(sum(abs(p_r_z - p_r_x))));

      if p_d_tx < p_d_zx
        z(i, j) = z(i, j) + delta;
        r_z(:, i : i + f_n1 - 1, j : j + f_n2 - 1) = p_r_t;

      end

    end
  end

end
