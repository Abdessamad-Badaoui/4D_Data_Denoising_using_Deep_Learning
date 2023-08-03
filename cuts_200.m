data_path = '/home/ics/Sqw/bnfs_10K.sqw';
data = sqw(data_path);
 
emode = 1; % direct geometry instrument
proj.u = [0,1,0];
proj.v = [0,0,1];
proj.type = 'rrr';
proj.uoffset = [0,0,0,0];
 
dq = 0.0125;
du = 0.0256495*0.5;
dv = 0.0319155*0.5;
 
w_min = min(data.data.p{1,3});
w_max = max(data.data.p{1,3});
 
e_min = min(data.data.p{1,4});
e_max = max(data.data.p{1,4});
 
dw = 2 * dq * [-1 1]; % slice thickness
E0 = 0;
dE = 0.12;
E = E0 + dE * [-1 1];
 
cutting_stepE = 2*dE;
cutting_stepEta = 4*dq;
 
for w= w_min:cutting_stepEta:w_max
    for e = e_min:cutting_stepE:e_max
        dw = [w w+4*dq]; % slice thickness
        E = [e e+2*dE];
        w1 = cut_sqw(data_path, proj, du, dv, dw, E, '-nopix');
        plot(w1);
        lz 0 1.5;
        % Convert in structure for save h5 format.
        ss = get(w1);
        aa.dat = ss.s;
        aa.x = 0.5 * (ss.p{1}(1:end-1) + ss.p{1}(2:end));
        aa.y = 0.5 * (ss.p{2}(1:end-1) + ss.p{2}(2:end));
        fname = sprintf('/home/ics/noise-tools-comparison/h5_files/bnfs_10K_E=%.2fmeV_eta=%.2f.h5', e+dE, w+2*dq);
        save (fname,'-struct','aa','-v7.3');
    end
end 

% Parametres à utiliser pour le merge :

disp('Les arguments à utiliser pour le merge :')
disp(['Number of cuts with respect to second dimension : ' num2str(round((e_max - e_min)/cutting_stepE))])
disp(['Number of cuts with respect to second dimension : ' num2str(round((w_max-w_min)/cutting_stepEta))])
disp(['Cutting step along the first dimension : ' num2str(cutting_stepE)])
disp(['Cutting step along the second dimension : ' num2str(cutting_stepEta)])
disp(['Min with respect to the first dimension : ' num2str(e_min)])
disp(['Min with respect to the second dimension : ' num2str(w_min)])
disp("Pour les deux dimesnions qui restent on ne peut pas les déduire directemnt, donc il suffit de voir les dimensions d'un exemple de coupe pour les savoir")





