
function z_struct = universal_jpeg_spp_vec_eval(x_path, y_path)

  f_n = 3;
  m = 4;

  x_struct = jpeg_read(x_path);
  x_coeffs = x_struct.coef_arrays{1};
  c_quant = x_struct.quant_tables{1};

  y_struct = jpeg_read(y_path);
  y_coeffs = y_struct.coef_arrays{1};

  z_coeffs = y_coeffs;

  fun = @(x) idct2(x.data .* c_quant);
  x_spatial = blockproc(x_coeffs, [8 8], fun);
  z_spatial = blockproc(z_coeffs, [8 8], fun);

  col_1 = im2col(x_spatial, [1, f_n], 'sliding');
  col_2 = im2col(x_spatial', [1, f_n], 'sliding');
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
    % conv2(base_f, base_f') / -1,
  };

  f_array_length = length(f_array);
  [f_n1, f_n2] = size(f_array{1});
  [n1, n2] = size(x_coeffs);

  r_x = zeros(f_array_length, n1 + f_n1 - 1, n2 + f_n2 - 1);
  r_z = zeros(f_array_length, n1 + f_n1 - 1, n2 + f_n2 - 1);
  for i = 1 : f_array_length
    f_item = f_array{i};
    r_x(i, :, :) = conv2(x_spatial, f_item, 'full');
    r_z(i, :, :) = conv2(z_spatial, f_item, 'full');
  end

  spatial_impact = cell(8, 8);
  for i = 1 : 8
    for j = 1 : 8
      tmp_coeffs = zeros(8, 8);
      tmp_coeffs(i, j) = m;
      spatial_impact{i, j} = idct2(tmp_coeffs .* c_quant);
    end
  end

  r_impact_n1 = 8 + f_n1 - 1;
  r_impact_n2 = 8 + f_n2 - 1;

  r_impact = cell(f_array_length, 8, 8);
  for i=1:8
    for j=1:8

      spatial_impact_item = spatial_impact{i, j};
      r_impact_item = zeros(f_array_length, r_impact_n1, r_impact_n2);
      for k = 1 : f_array_length
        f_item = f_array{k};
        r_impact_item(k, :, :) = conv2(spatial_impact_item, f_item, 'full');
      end

      r_impact{i, j} = r_impact_item;
    end
  end

  f = zeros(f_array_length, f_n1, f_n2);
  for i = 1 : f_array_length
    f(i, :, :) = f_array{i};
  end

  for i = 1 : n1
    for j = 1 : n2

      outer_i = (ceil(i / 8) - 1) * 8 + 1;
      outer_j = (ceil(j / 8) - 1) * 8 + 1;

      inner_i = mod(i - 1, 8) + 1;
      inner_j = mod(j - 1, 8) + 1;

      delta_sign = sign(x_coeffs(i, j) - y_coeffs(i, j));
      dct_delta = m * delta_sign;

      if (~delta_sign) || (z_coeffs(i, j) + dct_delta < -1024) || (z_coeffs(i, j) + dct_delta > 1023)
        continue;
      end

      p_r_x = r_x(:, outer_i : outer_i + r_impact_n1 - 1, outer_j : outer_j + r_impact_n2 - 1);
      p_r_z = r_z(:, outer_i : outer_i + r_impact_n1 - 1, outer_j : outer_j + r_impact_n2 - 1);

      r_impact_item = r_impact{inner_i, inner_j};

      p_r_t = p_r_z + r_impact_item * delta_sign;

      p_d_tx = sum(sum(sum(abs(p_r_t - p_r_x))));
      p_d_zx = sum(sum(sum(abs(p_r_z - p_r_x))));

      if p_d_tx < p_d_zx
        z_coeffs(i, j) = z_coeffs(i, j) + dct_delta;
        r_z(:, outer_i : outer_i + r_impact_n1 - 1, outer_j : outer_j + r_impact_n2 - 1) = p_r_t;

      end

    end
  end

  z_struct = y_struct;
  z_struct.coef_arrays{1} = z_coeffs;

end


