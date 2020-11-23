m = 3;
n = 2;

% [2 3]
% [1 2]
% [3 4]
A = [2 3; 1 2; 3 4];
constant_A = norm(2*transpose(A)*A);

% [1]
% [2]
% [3]
b = [1; 2; 3];
e = 1;

kappa = 0.2;

% p1 = [1 1] p2 = [2 0] p3 = [2 1]
%      [1 3]      [0 1]      [1 2]
p1 = [1 1; 1 3];
p2 = [2 0; 0 1];
p3 = [2 1; 1 2];
p1_i = inv(p1);
p2_i = inv(p2);
p3_i = inv(p3);
constant_p1 = norm(p1_i+transpose(p1_i));
constant_p2 = norm(p2_i+transpose(p2_i));
constant_p3 = norm(p3_i+transpose(p3_i));
% decide L and u
L = max([constant_A constant_p1 constant_p2 constant_p3]);
u = min([constant_A constant_p1 constant_p2 constant_p3]);

% initial step
% [0]
% [0]
x0 = [0.5; 0];
t0 = -100;
error = 0.1;

% set functions f0, f1,f2 and f3
syms x1 x2
x_sym = [x1; x2];
f0(x1, x2) = (norm(A*x_sym-b))^2;
f1(x1, x2) = (transpose(x_sym)*p1_i)*x_sym-1;
f2(x1, x2) = (transpose(x_sym)*p2_i)*x_sym-1;
f3(x1, x2) = (transpose(x_sym)*p3_i)*x_sym-1;

df0_x1(x1, x2) = diff(f0, x1);
df0_x2(x1, x2) = diff(f0, x2);
df1_x1(x1, x2) = diff(f1, x1);
df1_x2(x1, x2) = diff(f1, x2);
df2_x1(x1, x2) = diff(f2, x1);
df2_x2(x1, x2) = diff(f2, x2);
df3_x1(x1, x2) = diff(f3, x1);
df3_x2(x1, x2) = diff(f3, x2);

beta = (sqrt(L)-sqrt(u))/(sqrt(L)+sqrt(u));

t = t0;
global_stop = 0;

while 1
    k = 1;
    xk = x0;
    yk = xk;
    while 1
        ft0_1 = double(f0(xk(1), xk(2)));
        ft0_2 = double(df0_x1(xk(1), xk(2)));
        ft0_3 = double(df0_x2(xk(1), xk(2)));
        
        ft1_1 = double(f1(xk(1), xk(2)));
        ft1_2 = double(df1_x1(xk(1), xk(2)));
        ft1_3 = double(df1_x2(xk(1), xk(2)));
        
        ft2_1 = double(f2(xk(1), xk(2)));
        ft2_2 = double(df2_x1(xk(1), xk(2)));
        ft2_3 = double(df2_x2(xk(1), xk(2)));
        
        ft3_1 = double(f3(xk(1), xk(2)));
        ft3_2 = double(df3_x1(xk(1), xk(2)));
        ft3_3 = double(df3_x2(xk(1), xk(2)));
        
        [x_ft_u, ft_val_u] = fmincon(@(x) objective_ft(x, xk, t, u, ft0_1, ft0_2, ft0_3, ft1_1, ft1_2, ft1_3, ft2_1, ft2_2, ft2_3, ft3_1, ft3_2, ft3_3),x0,[],[],[],[],[],[],@(x) constraint_ft(x, e))
        [x_ft_L, ft_val_L] = fmincon(@(x) objective_ft(x, xk, t, L, ft0_1, ft0_2, ft0_3, ft1_1, ft1_2, ft1_3, ft2_1, ft2_2, ft2_3, ft3_1, ft3_2, ft3_3),x0,[],[],[],[],[],[],@(x) constraint_ft(x, e))
        
        if ft_val_L < error
            final_x = [x_ft_L];
            global_stop = 1;
            break;
        end
        
        x_ft_L_list_1(k) = x_ft_L(1)
        x_ft_L_list_2(k) = x_ft_L(2)
        ft_val_L_list(k) = ft_val_L
        
        if ft_val_u >= (1-kappa)*ft_val_L
            break;
        else
            fyk0_1 = double(f0(yk(1), yk(2)));
            fyk0_2 = double(df0_x1(yk(1), yk(2)));
            fyk0_3 = double(df0_x2(yk(1), yk(2)));
            
            fyk1_1 = double(f1(yk(1), yk(2)));
            fyk1_2 = double(df1_x1(yk(1), yk(2)));
            fyk1_3 = double(df1_x2(yk(1), yk(2)));
            
            fyk2_1 = double(f2(yk(1), yk(2)));
            fyk2_2 = double(df2_x1(yk(1), yk(2)));
            fyk2_3 = double(df2_x2(yk(1), yk(2)));
            
            fyk3_1 = double(f3(yk(1), yk(2)));
            fyk3_2 = double(df3_x1(yk(1), yk(2)));
            fyk3_3 = double(df3_x2(yk(1), yk(2)));
            
            [x_fyk, value_fyk] = fmincon(@(x) objective_fyk(x, yk, L, fyk0_1, fyk0_2, fyk0_3, fyk1_1, fyk1_2, fyk1_3, fyk2_1, fyk2_2, fyk2_3, fyk3_1, fyk3_2, fyk3_3), x0,[],[],[],[],[],[],@(x) constraint_ft(x, e))
            yk = x_fyk+beta*(x_fyk-xk)
            xk = x_fyk
            k = k+1            
        end        
    end
    
    
    if global_stop == 1
        break;
    end
    
    [kj_value, kj_index] = min(ft_val_L_list)
    x0 = [x_ft_L_list_1(kj_index); x_ft_L_list_2(kj_index)]
    
    x_last = [x_ft_L_list_1(length(x_ft_L_list_1)); x_ft_L_list_2(length(x_ft_L_list_2))]
    kj_value_last = ft_val_L_list(length(ft_val_L_list))
    
    ft_new_val_u = kj_value_last

    t_lower = t;
    t_upper = norm(b)^2;
    % if ft_new_val_u > 0
    %     t_lower = t0;
    % else
    %     t_upper = t0;
    % end
    
    while abs(ft_new_val_u) > 0.1
        t_mid = (t_lower+t_upper)/2;
                
        ft_new0_1 = double(f0(x_last(1), x_last(2)))
        ft_new0_2 = double(df0_x1(x_last(1), x_last(2)))
        ft_new0_3 = double(df0_x1(x_last(1), x_last(2)))
                
        ft_new1_1 = double(f1(x_last(1), x_last(2)))
        ft_new1_2 = double(df1_x1(x_last(1), x_last(2)))
        ft_new1_3 = double(df1_x1(x_last(1), x_last(2)))
                
        ft_new2_1 = double(f2(x_last(1), x_last(2)))
        ft_new2_2 = double(df2_x1(x_last(1), x_last(2)))
        ft_new2_3 = double(df2_x1(x_last(1), x_last(2)))
                
        ft_new3_1 = double(f3(x_last(1), x_last(2)))
        ft_new3_2 = double(df3_x1(x_last(1), x_last(2)))
        ft_new3_3 = double(df3_x1(x_last(1), x_last(2)))
        [x_ft_new_u, ft_new_val_u] = fmincon(@(x) objective_ft(x, x_last, t_mid, u, ft_new0_1, ft_new0_2, ft_new0_3, ft_new1_1, ft_new1_2, ft_new1_3, ft_new2_1, ft_new2_2, ft_new2_3, ft_new3_1, ft_new3_2, ft_new3_3),x0,[],[],[],[],[],[],@(x) constraint_ft(x, e))
        if ft_new_val_u > 0 
            t_lower = t_mid;
        else
            t_upper = t_mid;
        end
    end
    t = t_mid;
                
end

x_final = [final_x(1); final_x(2)]
disp("finish")



function [c,ceq] = constraint_ft(x, e)
c1 = norm([x(1); x(2)])-e;
c = [c1];
ceq = [];
end