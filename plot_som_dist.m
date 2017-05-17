function ed = plot_som_dist(sommap)

% Transform the 3D representation of neurons into 2D
neuronList = reshape(sommap,size(sommap,1)*size(sommap,2),size(sommap,3));               

% Initialize Euclidean Distance
ed = zeros(size(sommap,1)*size(sommap,2));

for i = 1:size(ed,1)
    for j = i:size(ed,1)
        ed(i,j) = sum((neuronList(i,:)- neuronList(j,:)).^2);
        ed(j,i) = ed(i,j);
    end
end
ed = sqrt(ed);
