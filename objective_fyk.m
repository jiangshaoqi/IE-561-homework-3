function fyk=objective_fyk(x, yk, L, fyk0_1, fyk0_2, fyk0_3, fyk1_1, fyk1_2, fyk1_3, fyk2_1, fyk2_2, fyk2_3, fyk3_1, fyk3_2, fyk3_3)
fyk0 = fyk0_1+[fyk0_2 fyk0_3]*[x(1)-yk(1);x(2)-yk(2)];
fyk1 = fyk1_1+[fyk1_2 fyk1_3]*[x(1)-yk(1);x(2)-yk(2)];
fyk2 = fyk2_1+[fyk2_2 fyk2_3]*[x(1)-yk(1);x(2)-yk(2)];
fyk3 = fyk3_1+[fyk3_2 fyk3_3]*[x(1)-yk(1);x(2)-yk(2)];
fyk_list = [fyk0 fyk1 fyk2 fyk3];
fyk = max(fyk_list)+L*norm([x(1)-yk(1);x(2)-yk(2)])^2/2;
end

