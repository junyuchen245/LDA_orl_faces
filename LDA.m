%% read all training images
imdata = [];

% go to every folder and get information
for folder_index = 1:40
    folder_index_str = num2str(folder_index);
    folder_location = strcat('orl_faces/Train/s',folder_index_str); 
    cd(folder_location);
    imdata_temp = zeros(9, 112*92);
    for file_index = 1:9
        imdata_temp_1 = imread(strcat(num2str(file_index),'.pgm'));
        imdata_temp(file_index,:) = reshape(imdata_temp_1, [1, 112*92]);
    end
    imdata = [imdata; imdata_temp];
    cd('../../../');
end

%% Centering the data
imdata = imdata';
mean_im = mean(imdata,2);
% remove mean
imdata = imdata - mean(imdata,2);
%% PCA on data
imdata_pca = my_PCA(imdata);
% obtain weighting
imdata_wts = imdata'*imdata_pca(:,1:300);
imdata_wts = imdata_wts';

%% read all testing images
testIm_data = [];
for folder_index = 1:40
    folder_index_str = num2str(folder_index);
    folder_location = strcat('orl_faces/Test/s',folder_index_str); 
    cd(folder_location);
    testIm_data_temp = zeros(1, 112*92);
    testIm_data_temp_1 = imread('10.pgm');
    testIm_data_temp(1,:) = reshape(testIm_data_temp_1, [1, 112*92]);
    testIm_data = [testIm_data; testIm_data_temp];
    cd('../../../');
end
testIm_data = testIm_data';
testIm_data = testIm_data - mean_im;
testIm_data_wts = testIm_data'*imdata_pca(:,1:300);
testIm_data_wts = testIm_data_wts';

%% calculate means for each class
m = mean(imdata_wts,2);
m_k = [];
for i = 1:9:360
    m_k_tmp = mean(imdata_wts(:,i:i+8),2);
    m_k = [m_k m_k_tmp];
end

%% compute S_b (between-class scatter)
S_b = 0;
for k = 1:40
    S_b_tmp = 9*(m_k(:,k) - m)*(m_k(:,k) - m)';
    S_b = S_b + S_b_tmp;
end

%% compute S_k and S_w (within-class scatter)
S_w = 0;
for k = 1:40
    S_k = 0;
    for i = 1:9
        S_k_tmp = (imdata_wts(:,(k-1)*9+i)-m_k(:,k))*(imdata_wts(:,(k-1)*9+i)-m_k(:,k))';
        S_k = S_k + S_k_tmp;
    end
    S_w = S_w + S_k;
end

%% LDA
L = 40;
[V, D] = eigs(S_b, S_w, L-1);

%% Classification with various dimensions for V
c = [];
d = [];
for dimension = 9:10:39
    if (dimension == 40)
        dimension = 39;
    end
prj_test = V(:,1:dimension)' * testIm_data_wts;
prj_train = V(:,1:dimension)' * imdata_wts;
% calculate correctness

correct = [];
for k = 1:40
    dist = [];
    dist_new = [];
    for i = 1:360
        % Euclidean distance
        dist_tmp = sum((prj_test(:,k)-prj_train(:,i)).^2);
        %dist_tmp = pdist2(prj_test(:,k)', prj_train(:,i)','euclidean'); %cosine distance
        dist = [dist dist_tmp];
    end
    [min_dist im_index] = min(dist);
    person = ceil(im_index/9);
    if(person == k)
        correct = [correct 1];
    else
        correct = [correct 0];
    end
end
correctness = length(find(correct == 1))/40;
disp(correctness)
c = [c correctness];
d = [d dimension];
end


%% PCA analysis
function projection = my_PCA(imdata)
% Covariance Matrix
cor_mat = imdata'*imdata;

% Eigen vector
[V,D] = eig(cor_mat);
egVal = diag(D);

% order by largest eigenvalues
egVal = egVal(end:-1:1);
V = V(:,end:-1:1);

%% project
projection = imdata*V;
end
