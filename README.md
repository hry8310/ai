<strong>AI概述</strong>
   <p> AI是自己学习开发的一些智能算法。其中dl主要是深度学习的内容（神经网络部分）。ml主要是机器学习的内容（包含部分opencv\dlib的视觉内容）
 
   </p>
   
   <p>
   <strong>ml</strong>
   <p> ml 目录主要包含一些机器学习的分类、聚类算法，当中混合了一些视觉相关的分类聚类算法（主要是依赖于opencv 和dlib 等库实现的小功能）
 
   <ul>
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
    <ul>
     <li>ex-face：opencv+dlib 实现的换脸。采用凸包和对应的特征点，划分多个三角形分别复制</li>
     <li>hear：使用opencv的Hear 工具，主要用于在liunx下生成各种sh脚本，包括样本create脚本、train 脚本等</li>
     <li>hog：使用opencv 的svm对图片的正负样本进行分类。对样本要求极高，特别是正样本，对于长高，角度，旋转都有严格的要求。相对于CNN的特征提取能力，相对较弱</li>
     <li>liveness：活体检测。包括摇头、张嘴、眨眼等</li>
     <li>sock_liveness：活体的sock版。通过网络传输视频帧实现</li>
     <li>same_face：检测同一张脸。在给定的多张人脸照片中，将是同一个人的照片归为一组，这样可以通过算法，将这些人的不同人脸照片。按人分组。
     	先是使用dlib提取人脸关键点特征。再根据聚类使用两种方式进行聚类：1、sklearn的DBSCAN，在聚类过程效果不太好，2、dlib的chinese_whispers。聚类效果明显优于 dbscan</li>
    </ul>
    
   </ul> 
   </p>
   
    
   <strong>dl </strong>
   <p> dl 目录主要包含一些深度学习的算法，主要是一些神经网络的实现，大多是cnn相关的神经网络
 
   <li>tcnn：居于tf（tensorflow,以后基本都用tf做简称）实现的cnn 文本分类器，主要使用区分文本相关内容，如：体育新闻、教育资讯等，使用简单的一层conv\pool+全连接+droopout。最后将输出层采用softmax做预测 </li>
   <li>dvc：居于tf实现的cnn 图片分类器，主要使用区分 cat 和 dog ，使用简单的三层conv\pool+全连接+droopout+全连接。最后将输出层做预测 </li>
   <li>rnn：居于tf 实现的简单rnn。里面有简单的lstm</li>
   <li>tf-yolo3 居于tf实现的yolo3，边框的损失都是使用平方损失函数，当中还包含了一些与移植相关的代码，如pb文件的生成和使用等</li>
   <br/><br/>
   
  
   <strong>持续更新中.....</strong>
