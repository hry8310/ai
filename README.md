
   <strong>AI概述</strong>
   <p> AI是自己学习开发的一些智能算法。其中dl主要是深度学习的内容（神经网络部分）。ml主要是机器学习的内容（包含部分opencv\dlib的视觉内容）
 
   </p>
   
   <p>
   <strong>ml</strong>
   <p> ml 目录主要包含一些机器学习的分类、聚类算法，当中混合了一些视觉相关的分类聚类算法（主要是依赖于opencv 和dlib 等库实现的小功能）
 
     
    <li>分类</li>
    <ul>
     <li>adaboost：python的原生实现</li>
     <li>bys：朴素贝叶斯的实现</li>
     <li>kmeans：实现最原始的kmeans，使用欧氏几何，支持高斯核（RBF）核函数映射到高维几何</li>
     <li>knn：python的原生实现，使用欧氏几何</li>
     <li>svm：python的原生实现，使用欧氏几何，支持多种核函数映射到高维几何，使用 cvxopt 解方程组。获得二次规划的解</li>
     <li>tree:实现ID3\C4.5\CART的版本。没有实现剪枝</li>
    </ul>
 
    
    <li>视觉</li>
    
    
  
   <p>
   
    
   <strong>dl </strong>
   <p> dl 目录主要包含一些深度学习的算法，主要是一些神经网络的实现，大多是cnn相关的神经网络
 
   <li>tcnn：居于tf（tensorflow,以后基本都用tf做简称）实现的cnn 文本分类器，主要使用区分文本相关内容，如：体育新闻、教育资讯等，使用简单的一层conv\pool+全连接+droopout。最后将输出层采用softmax做预测 </li>
   <li>dvc：居于tf实现的cnn 图片分类器，主要使用区分 cat 和 dog ，使用简单的三层conv\pool+全连接+droopout+全连接。最后将输出层做预测 </li>
   <li>rnn：居于tf 实现的简单rnn。里面有简单的lstm</li>
   <li>tf-yolo3 居于tf实现的yolo3，边框的损失都是使用平方损失函数，当中还包含了一些与移植相关的代码，如pb文件的生成和使用等</li>
   <br/><br/>
   
  
   <strong>持续更新中.....</strong>
