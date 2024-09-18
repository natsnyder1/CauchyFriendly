% This is the callback function correpsonding to the decription for point 2.) above 
function nonlinear_msmt_model(c_duc, c_zbar)
    H = [1.0, 0.0]; % meausrement model
    mduc = M_CauchyDynamicsUpdateContainer(c_duc);
    %% Set zbar
    xbar = mduc.cget_x(); % xbar
    zbar = H * xbar; % for other systems, call your nonlinear h(x) function
    mduc.cset_zbar(c_zbar, zbar);
end
