%% extract necessary variables (E_entry, y_classes)
energy_correlation;
y_classes = repmat(output_classes',3,1);
y_classes(:,1:time_sp) = repmat([1; 0; 0],1,time_sp);
y_classes(:,time_sp+1:time_pi) = repmat([0; 1; 0],1,time_pi-time_sp);
y_classes(:,time_pi+1:end) = repmat([0; 0; 1],1,size(y_classes(:,time_pi+1:end),2));

%% registry distance
d = 1000*ones(1,1663*1662/2);
index = 1;
for k=1:1663
    for l = k+1:1663
        d(index)= sum((E_entry(k,:)-E_entry(l,:)).^2,2);
        index = index+1;
    end
end

%% distance histogram
figure;
histogram(d)
figure;
histogram(d,100000)

%% Supervisoned Kohonen 1 (SKN)
n_classes = 3;
n_dim_entrada = size(E_entry,2);
W_kohonem = zeros(n_classes,n_dim_entrada);
epochs = 100;

F_in = zeros(n_classes, epochs);

W_kohonem(1,:) = E_entry(1,:);
W_kohonem(2,:) = E_entry(time_sp+1,:);
W_kohonem(3,:) = E_entry(time_pi+1,:);

W_kohonem_aux = [repmat(W_kohonem(1,:),time_sp,1);repmat(W_kohonem(2,:),time_pi-time_sp,1);...
    repmat(W_kohonem(3,:),size(E_entry(time_pi+1:end,1),1),1)];

baricenter = [mean(E_entry(1:time_sp,:),1);...
    mean(E_entry(time_sp+1:time_pi,:),1);...
    mean(E_entry(time_pi+1:end,:),1)];

alpha = 0.1;
x = E_entry;
bari_distance = zeros(3,epochs);
for e=1:epochs
   delta_kohonem = alpha*(x - W_kohonem_aux);
   
   F_in(:,e) = [sum(sum((delta_kohonem(1:time_sp,:)/alpha).^2,2),1);...
       sum(sum((delta_kohonem(time_sp+1:time_pi,:)/alpha).^2,2),1);...
       sum(sum((delta_kohonem(time_pi+1:end,:)/alpha).^2,2),1)];
   
   bari_distance(:,e) = sum((W_kohonem - baricenter).^2,2);
   
   W_kohonem = W_kohonem + [mean(delta_kohonem(1:time_sp,:),1); ...
       mean(delta_kohonem(time_sp+1:time_pi,:),1); mean(delta_kohonem(time_pi:end,:),1)];
   
   W_kohonem_aux = [repmat(W_kohonem(1,:),time_sp,1);repmat(W_kohonem(2,:),time_pi-time_sp,1);...
    repmat(W_kohonem(3,:),size(E_entry(time_pi+1:end,1),1),1)];
end

figure;
plot(F_in');
legend('SP','PE','PI')
xlabel('Epochs')
ylabel('Fin')

figure;
plot(sum(F_in,1));
xlabel('Epochs')
ylabel('Fin Total')

figure;
plot(bari_distance')
xlabel('Epochs')
ylabel('Distance to Baricenter')
legend('SP','PE','PI')

% ah = findobj('Type','figure'); % get all figures
% for m=1:numel(ah) % go over all axes
%   set(findall(ah(m),'-property','FontSize'),'FontSize',12)
%   axes_handle = findobj(ah(m),'type','axes');
%   ylabel_handle = get(axes_handle,'ylabel');
%   saveas(ah(m),[ylabel_handle.String '.png'])
% end
% 
% close all;

%% Supervisoned Kohonen 2 (SKN_1)
x = [E_entry(:,1:13) y_classes']';
net = selforgmap([10 10],400,2,'hextop','linkdist');
net = train(net,x);

figure
plotsomnd(net);

figure;
plotsomhits(net,x(:,1:time_sp))
title('SP Hits Extended')

figure;
plotsomhits(net,x(:,time_sp:time_pi))
title('PE Hits Extended')

figure;
plotsomhits(net,x(:,time_pi:end))
title('PI Hits Extended')

aux  = net.IW{1};
x_ = x(1:end-3,:);

net_ = selforgmap([10 10],100,2,'hextop','linkdist');
net_ = configure(net_,x_);

net_.IW{1} = aux(:,1:end-3);

figure;
plotsomhits(net_,x_(:,1:time_sp))
title('SP Hits')

figure;
plotsomhits(net_,x_(:,time_sp:time_pi))
title('PE Hits')

figure;
plotsomhits(net_,x_(:,time_pi:end))
title('PI Hits')

% ah = findobj('Type','figure'); % get all figures
% for m=1:numel(ah) % go over all axes
%   set(findall(ah(m),'-property','FontSize'),'FontSize',12)
%   axes_handle = findobj(ah(m),'type','axes');
%   saveas(ah(m),[axes_handle(1).Title.String '.png'])
% end
% close all;

w_input_fig_handle = figure;
   plotsomplanes(net)
   
   set(findall(w_input_fig_handle,'-property','FontSize'),'FontSize',12)
   axes_handle = findobj(w_input_fig_handle,'type','axes');
%   ylabel_handle = get(axes_handle,'ylabel');
for k=1:numel(axes_handle)
  axes_handle(k).Title.String = ['W' num2str(numel(axes_handle)-k)];
end
  saveas(w_input_fig_handle,[axes_handle(1).Title.String '.png'])

%% x-y fused map (SKN_2_X_Y_Fused)

[som som_y] = SOMSimple(10,300,E_entry(:,1:13), y_classes',0.15, 0.005, 30, 0.02, 1, 0.005);

% hits_sp = plot_som_hits(som, E_entry(1:time_sp,:));
% hits_pe = plot_som_hits(som, E_entry(time_sp:time_pi,:));
% hits_pi = plot_som_hits(som, E_entry(time_pi:end,:));
% figure;
% imagesc(hits_sp)
% 
% figure;
% imagesc(hits_pe)
% 
% figure;
% imagesc(hits_pi)

% distances = plot_som_dist(som);
% 
% distances = distances/max(max(distances));
% figure;
% imagesc(distances);

net = selforgmap([10 10],100,2,'gridtop','dist');
net = configure(net,E_entry(:,1:13)');

net_y = selforgmap([10 10],100,2,'gridtop','dist');
net_y = configure(net,y_classes);

% aux = aux';
% for k=1:5
%     aux_row = aux(k,:);
%     aux(k,:) = aux(11-k,:);
%     aux(11-k,:) = aux_row;
% end

% aux = aux';
net.IW{1} = reshape(som,size(som,1)*size(som,2),size(som,3));
net_y.IW{1} = reshape(som_y,size(som_y,1)*size(som_y,2),size(som_y,3));

figure
plotsomnd(net);

figure;
plotsomhits(net,E_entry(1:time_sp,1:13)')
title('SP Hits')

figure;
plotsomhits(net,E_entry(time_sp:time_pi,1:13)')
title('PE Hits')

figure;
plotsomhits(net,E_entry(time_pi:end,1:13)')
title('PI Hits')

% figure;
% plotsomnd(net_y);

% ah = findobj('Type','figure'); % get all figures
% for m=1:numel(ah) % go over all axes
%   set(findall(ah(m),'-property','FontSize'),'FontSize',12)
%   axes_handle = findobj(ah(m),'type','axes');
%   saveas(ah(m),[axes_handle(1).Title.String '.png'])
% end
% 
% 
% w_input_fig_handle = figure;
%    plotsomplanes(net,E_entry(1:13))
%    
%    set(findall(w_input_fig_handle,'-property','FontSize'),'FontSize',12)
%    axes_handle = findobj(w_input_fig_handle,'type','axes');
% %   ylabel_handle = get(axes_handle,'ylabel');
% for k=1:numel(axes_handle)
%   axes_handle(k).Title.String = ['W' num2str(numel(axes_handle)-k)];
% end
%   saveas(w_input_fig_handle,[axes_handle(1).Title.String '.png'])
  
%% unsupervisoned kohonen
x = E_entry(:,1:13)';
net = selforgmap([10 10],200,2,'hextop','linkdist');
net = train(net,x);

figure
plotsomnd(net);

figure;
plotsomhits(net,E_entry(1:time_sp,1:13)')
title('SP Hits')

figure;
plotsomhits(net,E_entry(time_sp:time_pi,1:13)')
title('PE Hits')

figure;
plotsomhits(net,E_entry(time_pi:end,1:13)')
title('PI Hits')


ah = findobj('Type','figure'); % get all figures
for m=1:numel(ah) % go over all axes
  set(findall(ah(m),'-property','FontSize'),'FontSize',12)
  axes_handle = findobj(ah(m),'type','axes');
  saveas(ah(m),[axes_handle(1).Title.String '.png'])
end


   w_input_fig_handle = figure;
   plotsomplanes(net,E_entry(1:13))
   
   
   set(findall(w_input_fig_handle,'-property','FontSize'),'FontSize',12)
   axes_handle = findobj(w_input_fig_handle,'type','axes');
%   ylabel_handle = get(axes_handle,'ylabel');
for k=1:numel(axes_handle)
  axes_handle(k).Title.String = ['W' num2str(numel(axes_handle)-k)];
end
  saveas(w_input_fig_handle,[axes_handle(1).Title.String '.png'])
  
 
