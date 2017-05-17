function hits = plot_som_hits(sommap, trainingData)

    hits = zeros(size(sommap,1),size(sommap,2));
    for ntraining = 1:size(trainingData,1)
        % Get current training vector
        trainingVector = trainingData(ntraining,:);
                
        % Compute the Euclidean distance between the training vector and
        % each neuron in the SOM map
        dist = getEuclideanDistance(trainingVector, sommap, size(sommap,1), size(sommap,1), size(sommap,3));
        
        % Find the best matching unit (bmu)
        [~, bmuindex] = min(dist);
        
        % transform the bmu index into 2D
        [bmurow, bmucol] = ind2sub([size(sommap,1) size(sommap,1)],bmuindex); 
        
        hits(bmurow,bmucol) = hits(bmurow,bmucol) +1;
    end


function ed = getEuclideanDistance(trainingVector, sommap, nrows, ncols, nfeatures)

% Transform the 3D representation of neurons into 2D
neuronList = reshape(sommap,nrows*ncols,nfeatures);               

% Initialize Euclidean Distance
ed = 0;
for n = 1:size(neuronList,2)
    ed = ed + (trainingVector(n)-neuronList(:,n)).^2;
end
ed = sqrt(ed);