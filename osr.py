import tensorflow as tf
import numpy as np
# a:Building the computational graph构建计算图
# b:Running the computational graph运行计算图
# 由两部分构成
class A1:
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)
    sess = tf.Session()
    def a1(self):
       print(self.node1,self.node2)
    def a2(self):  #要想打印最终结果，我们必须用到session:一个session封装了TensorFlow运行时的控制和状态
       print(self.sess.run([self.node1,self.node2]))
    def a3(self):
        a=tf.placeholder(tf.float32)
        b=tf.placeholder(tf.float32)
        adder_node=a+b
        print(self.sess.run(adder_node,{a:3,b:5}))
    def a4(self): #调用变量，使用前初始化
        w=tf.Variable([.3],dtype=tf.float32)
        b=tf.Variable([-.3],dtype=tf.float32)
        x=tf.placeholder(tf.float32)
        linerar_model=w*x+b
        init=tf.global_variables_initializer()
        self.sess.run(init)
        print(self.sess.run(linerar_model,{x:[1,2,3,4]}))
    def a5(self): #损失函数，对模型进行评估
        w = tf.Variable([.3], dtype=tf.float32)
        b = tf.Variable([-.3], dtype=tf.float32)
        x = tf.placeholder(tf.float32)
        linear_model = w * x + b
        y = tf.placeholder(tf.float32)
        squared_deltas = tf.square(linear_model - y)
        loss = tf.reduce_sum(squared_deltas)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print(self.sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
    #tf.assign(ref,value)把value值赋给ref
    def a6(self): #提供优化器，从而慢慢改变每一个变量而最小化损失函数,tf.train提供优化器
        # 重复
        w = tf.Variable([.3], dtype=tf.float32)
        b = tf.Variable([-.3], dtype=tf.float32)
        x = tf.placeholder(tf.float32)
        linear_model = w * x + b
        y = tf.placeholder(tf.float32)
        squared_deltas = tf.square(linear_model - y)
        loss = tf.reduce_sum(squared_deltas)
        # 重复
        optimizer=tf.train.GradientDescentOptimizer(0.01)
        train=optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for i in range(1000):
            self.sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
        print(self.sess.run([w,b]))
    def a7(self):
        labels = np.random.randint(12, size=(10, 2)) #随机生成随机数，最大值可省略，后面代表随机矩阵大小
        print(labels)

a22=A1()
a22.a7()