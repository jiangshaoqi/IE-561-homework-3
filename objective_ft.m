function ft=objective_ft(x, xk, t, uL, ft0_1, ft0_2, ft0_3, ft1_1, ft1_2, ft1_3, ft2_1, ft2_2, ft2_3, ft3_1, ft3_2, ft3_3)
ft0 = ft0_1+[ft0_2 ft0_3]*[x(1)-xk(1);x(2)-xk(2)]-t;
ft1 = ft1_1+[ft1_2 ft1_3]*[x(1)-xk(1);x(2)-xk(2)];
ft2 = ft2_1+[ft2_2 ft2_3]*[x(1)-xk(1);x(2)-xk(2)];
ft3 = ft3_1+[ft3_2 ft3_3]*[x(1)-xk(1);x(2)-xk(2)];
ft_list = [ft0 ft1 ft2 ft3];
ft = max(ft_list)+uL*norm([x(1)-xk(1);x(2)-xk(2)])^2/2;
end