% Veri setini yükleme
data = readtable('heart_disease_uci.csv'); % Veri setini yükleyin 

% 'sex' sütununu kategorik değişkene dönüştürme
data.sex = categorical(data.sex);

% 'cp' sütununu kategorik değişkene dönüştürme
data.cp = categorical(data.cp);

% 'restecg' sütununu kategorik değişkene dönüştürme
data.restecg = categorical(data.restecg);

% 'slope' sütununu kategorik değişkene dönüştürme
data.slope = categorical(data.slope);

% 'thal' sütununu kategorik değişkene dönüştürme
data.thal = categorical(data.thal);

% 'fbs' sütununu doğru tipe dönüştürme
data.fbs = data.fbs == "TRUE";

% 'exang' sütununu doğru tipe dönüştürme
data.exang = data.exang == "TRUE";

% Giriş özelliklerini ve hedef değişkeni seçme
inputs = [data.age, double(data.sex), double(data.cp), data.trestbps, data.chol, double(data.fbs), double(data.restecg), data.thalch, double(data.exang), data.oldpeak, double(data.slope), data.ca, double(data.thal)]'; % Giriş özellikleri
targets = data.num'; % Hedef çıktı

% Eğitim ve test setlerini oluşturma (örneğin, %70 eğitim, %30 test)
cv = cvpartition(size(data, 1), 'HoldOut', 0.3);
idxTrain = training(cv);
idxTest = test(cv);
inputsTrain = inputs(:, idxTrain);
targetsTrain = targets(idxTrain);
inputsTest = inputs(:, idxTest);
targetsTest = targets(idxTest);

% MLP modelini oluşturma
hiddenLayerSize = 10; % Gizli katman boyutu
net = patternnet(hiddenLayerSize); % MLP modelini oluşturun

% Modeli eğitme
[net,tr] = train(net,inputsTrain,targetsTrain);

% Eğitim seti üzerinde tahmin yapma ve doğruluk kontrolü
trainPredictions = net(inputsTrain);
trainAccuracy = sum(targetsTrain == round(trainPredictions)) / numel(targetsTrain);
disp(['Eğitim seti doğruluğu: ', num2str(trainAccuracy)]);

% Test seti üzerinde tahmin yapma
predictions = net(inputsTest);

% Tahminlerin doğruluğunu değerlendirme
testAccuracy = sum(targetsTest == round(predictions)) / numel(targetsTest);
disp(['Test seti doğruluğu: ', num2str(testAccuracy)]);

% Rastgele bir veri örneği seçme
randomIndex = randi(size(inputsTest, 2));

% Seçilen veri örneğini kullanarak tahmin yapma
inputSample = inputsTest(:, randomIndex);
targetSample = targetsTest(randomIndex);
prediction = round(net(inputSample));

% Tahmin sonucunu gösterme
disp(['Girişler: ', num2str(inputSample')]); % 'inputSample'ı transpoze edin
disp(['Gerçek Çıktı: ', num2str(targetSample)]);
disp(['Tahmin: ', num2str(prediction)]);





