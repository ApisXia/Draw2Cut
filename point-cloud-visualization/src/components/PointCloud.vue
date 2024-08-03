<template>
  <div>
    <h1>Drawn2Cut interface</h1>
    <div class="container">
      <div class="left-panel">
        <canvas ref="canvas"></canvas>
        <button @click="autoSmooth">Auto-smooth</button>
      </div>
      <div class="right-panel">
        <div v-for="text in texts" :key="text">{{ text }}</div>
      </div>
    </div>
  </div>
</template>

<script>
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';


export default {
  data() {
    return {
      texts: []
    };
  },
  methods: {
    fetchPointCloudData() {
      fetch('http://localhost:5000/data')
        .then(response => response.json())
        .then(data => {
          this.renderPointCloud(data);
          this.texts = data.texts;
        });
    },
    renderPointCloud(data) {
      // 创建场景
      var scene = new THREE.Scene();
      var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

      // 创建WebGLRenderer并设置背景颜色为白色
      var renderer = new THREE.WebGLRenderer();
      // renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setSize(300,150);
      renderer.setPixelRatio(0.5);
      // renderer.setClearColor(0xffffff);  // 设置背景颜色为白色
      this.$refs.canvas.appendChild(renderer.domElement);

      // 创建OrbitControls
      var controls = new OrbitControls(camera, renderer.domElement);

      var wood_points = data.wood_points;
      var wood_colors = data.wood_colors;

      // 创建木头的点云
      var wood_geometry = new THREE.BufferGeometry();
      var wood_vertices = new Float32Array(wood_points.length * 3);
      var wood_vertexColors = new Float32Array(wood_points.length * 3);

      for (var i = 0; i < wood_points.length; i++) {
        wood_vertices[i * 3] = wood_points[i][0];
        wood_vertices[i * 3 + 1] = wood_points[i][1];
        wood_vertices[i * 3 + 2] = wood_points[i][2];

        wood_vertexColors[i * 3] = wood_colors[i][0];
        wood_vertexColors[i * 3 + 1] = wood_colors[i][1];
        wood_vertexColors[i * 3 + 2] = wood_colors[i][2];
      }

      wood_geometry.setAttribute('position', new THREE.BufferAttribute(wood_vertices, 3));
      wood_geometry.setAttribute('color', new THREE.BufferAttribute(wood_vertexColors, 3));

      var wood_material = new THREE.PointsMaterial({ size: 0.05, vertexColors: true });
      var wood_pointCloud = new THREE.Points(wood_geometry, wood_material);
      scene.add(wood_pointCloud);

      camera.position.z = 5;

      var animate = function () {
        requestAnimationFrame(animate);
        controls.update();  // 更新OrbitControls
        renderer.render(scene, camera);
      };

      animate();

      // 响应窗口大小变化
      window.addEventListener('resize', function () {
        var width = window.innerWidth;
        var height = window.innerHeight;
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      });
    },
    autoSmooth() {
      fetch('http://localhost:5000/auto-smooth', {
        method: 'POST'
      })
        .then(response => response.json())
        .then(data => {
          this.renderPointCloud(data);
          this.texts = data.texts;
        });
    }
  },
  mounted() {
    this.fetchPointCloudData();
  }
};
</script>

<style>
.container {
  display: flex;
  height: 100%;
}

.left-panel {
  flex: 2;
  position: relative;
}

.right-panel {
  flex: 1;
  padding: 20px;
  background-color: #f0f0f0;
}
</style>
