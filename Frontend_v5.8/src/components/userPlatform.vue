<template>
  <!--
  更新页面 --v5.6
  版本  2024-7-19
  -->
  <div style="height: 100vh; overflow: hidden" @mouseover="background_IMG">
    <el-container class="fullscreen_container">
      <el-header
        style="height: 60px; text-align: center; line-height: 60px; position: relative"
      >
        <img
          src="./assets/logo.png"
          alt=""
          style="width: 50px; position: absolute; left: 5px; top: 5px"
        />
        <h2 style="font-size: 26px">车轮状态分析与健康评估</h2>
        <div
          class="user-info-container"
          id="userInfo"
          style="position: absolute; right: 10px; top: 5px; color: white"
        >
          <span style="margin-right: 10px">欢迎！ {{ username }}</span>
          <span @click="logout" class="clickable">退出登录</span>
        </div>
      </el-header>
      <el-container>
        <el-aside width="250px">
          <div
            style="
              font-size: 20px;
              font-weight: 700;
              background-color: #80a5ba;
              width: 250px;
              color: #f9fbfa;
            "
          >
            算法选择区
          </div>
          <div style="background-color: #eff3f6; width: 250px; height: 600px">
            <el-scrollbar height="600px" min-size="35" style="margin-left: 10px">
              <el-column v-for="item in menuList2">
                <el-row
                  ><el-button
                    style="
                      width: 150px;
                      margin-top: 10px;
                      background-color: #4599be;
                      color: white;
                    "
                    icon="ArrowDown"
                    @click="
                      menu_details_second[item.label] = !menu_details_second[item.label]
                    "
                  >
                    <el-text
                      style="width: 105px; font-size: 15px; color: white"
                      truncated
                      >{{ item.label }}</el-text
                    ></el-button
                  ></el-row
                >

                <el-column
                  v-if="menu_details_second[item.label]"
                  v-for="option in item.options"
                >
                  <el-row style="margin-left: 20px">
                    <el-button
                      style="width: 150px; margin-top: 7px; background-color: #75acc3"
                      icon="ArrowDown"
                      type="info"
                      @click="
                        menu_details_third[option.label] = !menu_details_third[
                          option.label
                        ]
                      "
                    >
                      <el-text
                        style="width: 105px; font-size: 12px; color: white"
                        truncated
                        >{{ option.label }}</el-text
                      ></el-button
                    >
                  </el-row>
                  <el-column
                    v-if="menu_details_third[option.label]"
                    v-for="algorithm in Object.keys(option.parameters)"
                  >
                    <el-tooltip
                      placement="right-start"
                      :content="labels_for_algorithms[algorithm]"
                      effect="light"
                    >
                      <div
                        :draggable="true"
                        @dragend="handleDragend($event, algorithm, option)"
                        class="item"
                        @click="showIntroduction(algorithm.replace(/_multiple/g, ''))"
                        style="
                          background-color: #f9fcff;
                          margin-top: 7px;
                          width: 145px;
                          height: 30px;
                          margin-bottom: 10px;
                          padding: 0px;
                          border-radius: 5px;
                          align-content: center;
                          margin-left: 40px;
                        "
                      >
                        <el-text style="width: 105px; font-size: 12px" truncated>{{
                          labels_for_algorithms[algorithm]
                        }}</el-text>
                      </div>
                    </el-tooltip>
                  </el-column>
                </el-column>
              </el-column>
            </el-scrollbar>
          </div>
          <div
            style="
              font-size: 20px;
              font-weight: 700;
              background-color: #80a5ba;
              width: 250px;
              color: #f9fbfa;
            "
          >
            加载数据
          </div>
          <uploadDatafile @switchDrawer="handleSwitchDrawer" :api="api" />
        </el-aside>

        <el-main
          @dragover.prevent
          ref="efContainerRef"
          id="efContainer "
          style="height: auto; width: 600px; padding: 0px"
        >
          <div
            style="
              position: relative;
              height: 25%;
              font-size: 20px;
              color: #003e50;
              font-weight: 500;
              font-family: Arial, Helvetica, sans-serif;
              background-position: center;
            "
          >
            <div id="statusIndicator" class="status-indicator">未建立模型</div>
            <DraggableContainer :referenceLineVisible="false">
            <Vue3DraggableResizable
              :draggable="true"
              :resizable="false"
              v-for="(item, index) in nodeList"
              :initH="90"
              :initW="123"
              
              :key="item.nodeId"
              class="node-info node-info-label"
              @drag-end="recordPosition(item, $event)"
              @click="resultShow(item)"
            >
              
                
                  {{ item.label_display }}
                  <!-- <div
                    style="
                      position: absolute;
                      left: 55px;
                      top: 35px;
                      width: 6px;
                      height: 6px;
                      border: 2px solid #80a5ba; /* 边框颜色*/
                      border-radius: 50%; /* 设置为50%以创建圆形 */
                      background-color: transparent; /* 背景设置为透明，实现空心效果 */
                      /* 其他样式，如 cursor 可以设置拖拽时的鼠标光标形状 */
                      cursor: move; /* 鼠标悬停时显示可移动的光标 */
                    "
                  ></div> -->
                  <el-button
                    type="danger"
                    icon="Delete"
                    circle
                    size="small"
                    class="deleteButton"
                    @click="deleteNode(item.nodeId)"
                    :disabled="modelSetup"
                  />
                  <div class="node-drag" :id="item.id"></div>
                  
                
              
            </Vue3DraggableResizable>
            </DraggableContainer>
            <div
              style="
                position: absolute;
                right: 17px;
                bottom: 20px;
                width: 1000px;
                height: auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
              "
            >
              <el-space size="large">
                <el-button
                  type="info"
                  round
                  style="width: 125px; font-size: 17px; background-color: #606266"
                  @click="fetch_models"
                  icon="More"
                >
                  历史模型
                </el-button>
                <!-- <el-upload action="" :auto-upload="false" v-model:file-list="file_list" :before-upload="checkFileType"
                          limit="1" >
                    <el-button type="info" round style="position: absolute; width: 125px; font-size: 17px; "  icon="FolderAdd">
                      上传数据
                    </el-button>
                </el-upload> -->

                <el-button
                  type="info"
                  round
                  style="width: 125px; font-size: 17px; background-color: #e6a23c"
                  @click="handleclear"
                  icon="Refresh"
                >
                  清空模型
                </el-button>

                <el-button
                  v-if="!to_rectify_model"
                  type="primary"
                  :disabled="canCompleteModeling"
                  @mouseover="CompleteModeling"
                  round
                  style="width: 125px; font-size: 17px"
                  @click="finished_model"
                  icon="Check"
                >
                  完成建模
                </el-button>

                <el-button
                  v-if="to_rectify_model"
                  type="primary"
                  :disabled="canCompleteModeling"
                  @mouseover="CompleteModeling"
                  round
                  style="width: 125px; font-size: 17px"
                  @click="rectify_model"
                  icon="Edit"
                >
                  修改模型
                </el-button>

                <!-- <el-button-group>
                  <el-button type="info" :disabled="canCheckModel" @mouseover="checkModeling" round style="width: 125px; font-size: 17px; background-color: #4ca1c1;" @click="check_model" icon="Search">
                    检查模型
                  </el-button>
                  <el-button type="info" round style="width: 70px; font-size: 17px; background-color: #4ca1c1;" icon="Opportunity">提示</el-button>
                </el-button-group> -->
                <!-- <el-button type="info" :disabled="canCheckModel" @mouseover="checkModeling" round
                  style="width: 125px; font-size: 17px; background-color: #4ca1c1;" @click="check_model" icon="Search">
                  检查模型
                </el-button> -->
                <el-button
                  type="primary"
                  :disabled="canCheckModel"
                  @mouseover="checkModeling"
                  round
                  style="width: 125px; font-size: 17px"
                  @click="check_model"
                  icon="Search"
                >
                  检查模型
                </el-button>

                <!-- <el-button type="info" :disabled="canSaveModel" @mouseover="saveModeling" round
                  style="width: 125px; font-size: 17px; background-color: #20A0FF;" @click="saveModelSetting(true, [])"
                  icon="Finished">
                  保存模型
                </el-button> -->
                <el-button
                  type="primary"
                  :disabled="canSaveModel"
                  @mouseover="saveModeling"
                  round
                  style="width: 125px; font-size: 17px"
                  @click="saveModelSetting(true, [])"
                  icon="Finished"
                >
                  保存模型
                </el-button>
                <!-- <el-button type="success" round style="width: 125px; font-size: 17px; background-color: #67C23A;"
                  @click="handleupload" icon="SwitchButton" :disabled="canStartProcess" @mouseover="startModeling">
                  开始运行
                </el-button> -->
                <el-button
                  type="success"
                  round
                  style="width: 125px; font-size: 17px"
                  @click="run"
                  icon="SwitchButton"
                  :disabled="canStartProcess"
                  @mouseover="startModeling"
                >
                  开始运行
                </el-button>
                <el-button
                  :disabled="canShutdown"
                  type="danger"
                  round
                  style="width: 125px; font-size: 17px"
                  @click="shutdown"
                  icon="Close"
                >
                  终止运行
                </el-button>
              </el-space>
            </div>
          </div>

          <div class="test" style="background-color: white">
            <!--            <div-->
            <!--              style="position: relative; text-align: center; color: #333333; font-size: 20px; background-color: #333333; width: 100%; height: 22px;font-family:Arial, Helvetica, sans-serif; font-weight: 500;"-->
            <!--              v-loading="loading">-->
            <el-dialog
              v-model="dialogVisible"
              title="模型算法及参数设置"
              style="width: 1000px; height: 750px"
            >
              <el-tabs v-model="activeName" class="demo-tabs" @tab-click="handleClick">
                <el-tab-pane
                  v-for="item in nodeList"
                  :label="item.label"
                  :name="item.nodeId"
                >
                  <div
                    v-if="item.label == '层次分析模糊综合评估'"
                    style="position: relative; height: 630px; width: auto"
                  >
                    <div
                      style="
                        position: absolute;
                        left: 10px;
                        top: 10px;
                        height: 480px;
                        width: 200px;
                        background-color: aliceblue;
                      "
                    >
                      <el-text size="large">指标层次构建</el-text>
                      <el-tree
                        style="max-width: 600px"
                        :data="dataSource"
                        show-checkbox
                        node-key="id"
                        default-expand-all
                        :expand-on-click-node="false"
                      >
                        <template #default="{ node, data }">
                          <span class="custom-tree-node">
                            <span>{{ node.label }}</span>
                            <span>
                              <a @click="append(data)"> Append </a>
                              <a style="margin-left: 8px" @click="remove(node, data)">
                                Delete
                              </a>
                            </span>
                          </span>
                        </template>
                      </el-tree>
                    </div>
                    <div
                      style="
                        position: absolute;
                        left: 230px;
                        top: 10px;
                        height: 480px;
                        width: 720px;
                        background-color: aliceblue;
                      "
                    >
                      <el-text size="large">指标权重配置</el-text>

                      <div
                        style="
                          position: absolute;
                          top: 25px;
                          left: 10px;
                          width: 500px;
                          height: 400px;
                          background-color: white;
                        "
                      ></div>
                      <div
                        style="
                          position: absolute;
                          top: 25px;
                          left: 500px;
                          width: 190px;
                          height: 400px;
                          background-color: lightgray;
                          margin-left: 20px;
                        "
                      >
                        <el-space direction="vertical">
                          <el-text style="font-weight: bold; font-size: larger"
                            >分值对照表</el-text
                          >
                          <el-row
                            ><el-text style="font-size: medium"
                              >1(1)同样重要</el-text
                            ></el-row
                          >
                          <el-row
                            ><el-text style="font-size: medium"
                              >2(1/2)稍稍微(不)重要</el-text
                            ></el-row
                          >
                          <el-row
                            ><el-text style="font-size: medium"
                              >3(1/3)稍微(不)重要</el-text
                            ></el-row
                          >
                          <el-row
                            ><el-text style="font-size: medium"
                              >4(1/4)稍比较(不)重要</el-text
                            ></el-row
                          >
                          <el-row
                            ><el-text style="font-size: medium"
                              >5(1/5)比较(不)重要</el-text
                            ></el-row
                          >
                          <el-row
                            ><el-text style="font-size: medium"
                              >6(1/6)稍非常(不)重要</el-text
                            ></el-row
                          >
                          <el-row
                            ><el-text style="font-size: medium"
                              >7(1/7)非常(不)重要</el-text
                            ></el-row
                          >
                          <el-row
                            ><el-text style="font-size: medium"
                              >8(1/8)稍绝对(不)重要</el-text
                            ></el-row
                          >
                          <el-row
                            ><el-text style="font-size: medium"
                              >9(1/9)绝对(不)重要</el-text
                            ></el-row
                          >
                        </el-space>
                      </div>
                    </div>
                    <div
                      style="
                        position: absolute;
                        left: 10px;
                        top: 500px;
                        width: 940px;
                        height: 125px;
                        background-color: aliceblue;
                      "
                    ></div>
                  </div>
                  <!-- 选择具体算法以及设置参数 -->
                  <div
                    v-if="item.label != '层次分析模糊综合评估'"
                    style="margin-top: 5px; margin-left: 30px; width: 550px"
                  >
                    <el-row>
                      <el-col :span="8"
                        ><el-text style="margin-top: 20px">选择算法</el-text></el-col
                      >
                      <el-col :span="16">
                        <el-select
                          v-model="item.use_algorithm"
                          placeholder="算法选择"
                          popper-append-to-body="false"
                          id="select_algorithm"
                          @change="setParams"
                        >
                          <el-option
                            v-for="(value, key) in item.parameters"
                            :label="labels_for_algorithms[key]"
                            :value="key"
                            style="width: 225px; background-color: white; height: auto"
                          >
                          </el-option>
                        </el-select>
                      </el-col>
                    </el-row>
                    <!-- 特征提取选择要显示的特征 -->
                    <el-row v-if="item.id == '1.3'">
                      <el-col :span="8">选择特征</el-col>
                      <el-col :span="16">
                        <div class="m-4">
                          <el-select
                            v-model="features"
                            multiple
                            collapse-tags
                            collapse-tags-tooltip
                            placeholder="选择需要提取的特征"
                          >
                            <el-option
                              v-for="(value, key) in item.parameters[item.use_algorithm]"
                              :label="key"
                              :value="key"
                              style="width: 200px; background-color: white"
                            />
                          </el-select>
                        </div>
                      </el-col>
                    </el-row>
                    <el-row>
                      <el-text
                        v-if="item.use_algorithm"
                        style="margin-top: 1px; margin-bottom: 3px"
                      >
                        算法介绍：{{ algorithm_introduction[item.use_algorithm] }}
                      </el-text>
                    </el-row>
                    <!-- 填参数 -->
                    <el-row
                      v-if="item.use_algorithm != null && item.id != '1.3'"
                      v-for="(value, key) in item.parameters[item.use_algorithm]"
                      :key="item.parameters[item.use_algorithm].keys"
                    >
                      <el-col :span="8"
                        ><span>{{ labels_for_params[key] }}</span></el-col
                      >
                      <el-col :span="16"
                        ><el-input
                          style="width: 190px"
                          :disabled="false"
                          v-model="item.parameters[item.use_algorithm][key]"
                      /></el-col>
                    </el-row>
                  </div>
                </el-tab-pane>
              </el-tabs>
            </el-dialog>
            <!--            </div>-->

            <!-- 点击展示算法的具体介绍 -->
            <!-- <span style="width: auto; height: auto;">
              <el-text v-if="showPlainIntroduction" :v-model="introduction" type="success" size="large">{{ introduction }}</el-text>
            </span> -->
            <v-md-preview
              v-if="showPlainIntroduction"
              :text="introduction_to_show"
              style="text-align: left"
            ></v-md-preview>
            <v-md-preview
              v-if="showStatusMessage"
              :text="status_message_to_show"
              style="text-align: center"
            ></v-md-preview>
            <!-- <el-table id="progress" :v-loading="loading" element-loading-text="正在处理" :data="tableData" style="width: 100%; height: 250px ;" v-if="loading">
            </el-table> -->
            <!-- <span  :v-loading="loading" element-loading-text="正在处理" element-loading-background="rgba(122, 122, 122, 0.8)" style="width: 100%; height: 100%;"></span> -->
            <el-progress
              v-if="processing"
              :percentage="percentage"
              :indeterminate="true"
            />
            <!-- <iframe
              id="my_gradio_app"
              style="width: 1200px; height: 570px"
              :src="refreshPardon"
              frameborder="0"
              v-if="isshow"
            >
            </iframe> -->

            <el-scrollbar height="600px" v-if="result_show">
              <!-- 健康评估可视化 -->
              <el-tabs
                class="demo-tabs"
                type="card"
                v-model="activeName1"
                v-if="display_health_evaluation"
              >
                <el-tab-pane label="层级有效指标" name="first">
                  <img
                    :src="health_evaluation_figure_1"
                    alt="figure1"
                    id="health_evaluation_figure_1"
                    class="result_image"
                    style="width: 900px; height: 450px"
                  />
                </el-tab-pane>
                <el-tab-pane label="指标权重" name="second">
                  <img
                    :src="health_evaluation_figure_2"
                    alt="figure2"
                    id="health_evaluation_figure_2"
                    class="result_image"
                    style="width: 900px; height: 450px"
                  />
                </el-tab-pane>
                <el-tab-pane label="评估结果" name="third">
                  <el-col>
                    <img
                      :src="health_evaluation_figure_3"
                      alt="figure3"
                      id="health_evaluation_figure_3"
                      class="result_image"
                      style="width: 900px; height: 450px"
                    />
                    <br />
                    <div style="width: 1000px; margin-left: 250px">
                      <el-text
                        :v-model="health_evaluation"
                        style="font-weight: bold; font-size: 18px"
                        >{{ health_evaluation }}</el-text
                      >
                    </div>
                  </el-col>
                </el-tab-pane>
              </el-tabs>
              <!-- 特征提取可视化 -->
              <el-table
                :data="transformedData"
                style="width: 100%; margin-top: 20px"
                v-if="display_feature_extraction"
              >
                <el-table-column
                  v-for="column in columns"
                  :key="column.prop"
                  :prop="column.prop"
                  :label="column.label"
                  :width="column.width"
                >
                </el-table-column>
              </el-table>
              <!-- 特征选择可视化 -->
              <div v-if="display_feature_selection">
                <img
                  :src="feature_selection_figure"
                  alt="feature_selection_figure1"
                  class="result_image"
                  style="width: 900px; height: 450px"
                />
                <br />
                <div style="width: 1000px; margin-left: 250px">
                  <el-text
                    :v-model="features_selected"
                    style="font-weight: bold; font-size: 20px"
                    >选取特征结果为： {{ features_selected }}</el-text
                  >
                </div>
              </div>
              <!-- 故障诊断可视化 -->
              <div
                v-if="display_fault_diagnosis"
                style="margin-top: 20px; font-size: 18px"
              >
                <div style="width: 1000px; margin-left: 250px; font-weight: bold">
                  故障诊断结果为： 由输入的振动信号，根据故障诊断算法得知该部件<span
                    :v-model="fault_diagnosis"
                    style="font-weight: bold; color: red"
                    >{{ fault_diagnosis }}</span
                  >
                </div>

                <br />
                <img
                  :src="fault_diagnosis_figure"
                  alt="fault_diagnosis_figure"
                  class="result_image"
                  style="width: 900px; height: 580px"
                />
              </div>
              <!-- 故障趋势预测可视化 -->
              <div
                v-if="display_fault_regression"
                style="margin-top: 20px; font-size: 18px"
              >
                <div style="width: 1000px; margin-left: 250px; font-weight: bold">
                  经故障诊断算法，目前该部件<span
                    :v-model="fault_regression"
                    style="font-weight: bold; color: red"
                    >{{ fault_regression }}</span
                  >
                  <span
                    v-if="!have_fault"
                    :v-model="time_to_fault"
                    style="font-weight: bold"
                    >根据故障预测算法预测，该部件{{ time_to_fault }}后会出现故障</span
                  >
                </div>
                <br />
                <img
                  :src="fault_regression_figure"
                  alt="fault_regression_figure"
                  class="result_image"
                  style="width: 900px; height: 580px"
                />
              </div>
              <!-- 插值处理结果可视化 -->
              <div v-if="display_interpolation" style="margin-top: 20px; font-size: 18px">
                <br />
                <img
                  :src="interpolation_figure"
                  alt="interpolation_figure"
                  class="result_image"
                  style="width: 900px; height: 450px"
                />
              </div>
            </el-scrollbar>
          </div>
        </el-main>

        <!-- 以抽屉的形式打开功能区 -->

        <el-drawer v-model="models_drawer" direction="ltr">
          <div style="display: flex; flex-direction: column">
            <el-col>
              <h2 style="margin-bottom: 25px; color: #253b45">历史模型</h2>

              <el-table :data="fetchedModelsInfo" height="500" stripe style="width: 100%">
                <el-popover
                  placement="bottom-start"
                  title="模型信息"
                  :width="400"
                  trigger="hover"
                >
                </el-popover>
                <el-table-column :width="100" property="id" label="序号" />
                <el-table-column :width="150" property="model_name" label="模型名称" />
                <el-table-column :width="280" label="操作">
                  <template #default="scope">
                    <el-button
                      size="small"
                      type="primary"
                      style="width: 50px"
                      @click="use_model(scope.row)"
                    >
                      使用
                    </el-button>
                    <el-button
                      size="small"
                      type="danger"
                      style="width: 50px"
                      @click="delete_model(scope.$index, scope.row)"
                    >
                      删除
                    </el-button>
                    <el-popover placement="bottom" :width="500" trigger="click">
                      <el-descriptions
                        :title="model_name"
                        :column="3"
                        :size="size"
                        direction="vertical"
                      >
                        <el-descriptions-item label="使用模块" :span="3">
                          <el-tag size="small" v-for="algorithm in model_algorithms">{{
                            algorithm
                          }}</el-tag>
                        </el-descriptions-item>
                        <el-descriptions-item label="算法参数" :span="3">
                          <div v-for="item in model_params">
                            {{ item.模块名 }}: {{ item.算法 }}
                          </div>
                        </el-descriptions-item>
                      </el-descriptions>
                      <template #reference>
                        <el-button
                          size="small"
                          type="info"
                          style="width: 80px"
                          @click="show_model_info(scope.row)"
                        >
                          查看模型
                        </el-button>
                      </template>
                    </el-popover>
                  </template>
                </el-table-column>
              </el-table>

              <el-dialog v-model="delete_model_confirm_visible" title="提示" width="500">
                <span style="font-size: 20px">确定删除该模型吗？</span>
                <template #footer>
                  <el-button
                    style="width: 150px"
                    @click="delete_model_confirm_visible = false"
                    >取消</el-button
                  >
                  <el-button
                    style="width: 150px; margin-right: 70px"
                    type="primary"
                    @click="delete_model_confirm"
                    >确定</el-button
                  >
                </template>
              </el-dialog>
            </el-col>
          </div>
        </el-drawer>

        <!-- 以抽屉的形式打开用户历史数据 -->
        <el-drawer v-model="data_drawer" direction="ltr">
          <div style="display: flex; flex-direction: column">
            <el-col>
              <h2 style="margin-bottom: 25px; color: #253b45">用户数据文件</h2>

              <el-table :data="fetchedDataFiles" height="500" stripe style="width: 100%">
                <!-- <el-table-column :width="100" property="id" label="文件序号" /> -->
                <el-table-column :width="150" property="dataset_name" label="文件名称" />
                <el-table-column :width="200" property="description" label="文件描述" />
                <el-table-column :width="200" label="操作">
                  <template #default="scope">
                    <el-button
                      size="small"
                      type="primary"
                      style="width: 50px"
                      @click="use_dataset(scope.row)"
                    >
                      使用
                    </el-button>
                    <el-button
                      size="small"
                      type="danger"
                      style="width: 50px"
                      @click="delete_dataset(scope.$index, scope.row)"
                    >
                      删除
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>

              <el-dialog
                v-model="delete_dataset_confirm_visible"
                title="提示"
                width="500"
              >
                <span style="font-size: 20px">确定删除该数据文件吗？</span>
                <template #footer>
                  <el-button
                    style="width: 150px"
                    @click="delete_dataset_confirm_visible = false"
                    >取消</el-button
                  >
                  <el-button
                    style="width: 150px; margin-right: 70px"
                    type="primary"
                    @click="delete_dataset_confirm"
                    >确定</el-button
                  >
                </template>
              </el-dialog>

              <!-- <el-dialog v-model="dialogFormVisible" title="保存模型" draggable width="40%">
                <el-form :model="model_info_form">
                  <el-form-item label="模型名称" :label-width='140'>
                    <el-input style="width: 160px;" v-model="model_info_form.name" autocomplete="off" />
                  </el-form-item>
                </el-form>
                <span class="dialog-footer">
                  <el-button style="margin-left: 400px; width: 150px;" @click="dialogFormVisible = false">取消</el-button>
                  <el-button style="width: 150px;" type="primary" @click="save_model_confirm">确定</el-button>
                </span>
              </el-dialog> -->
            </el-col>
          </div>
        </el-drawer>
      </el-container>
      <el-footer style="height: 35px; position: relative"
        ><span style="position: absolute; top: -15px">welcome!</span></el-footer
      >
    </el-container>
    <el-dialog v-model="dialogmodle" title="保存模型" draggable width="30%">
      <el-form :model="model_info_form">
        <el-form-item label="模型名称" :label-width="140">
          <el-input
            style="width: 160px"
            v-model="model_info_form.name"
            autocomplete="off"
          />
        </el-form-item>
      </el-form>
      <span class="dialog-footer">
        <el-button style="margin-left: 85px; width: 150px" @click="dialogmodle = false"
          >取消</el-button
        >
        <el-button style="width: 150px" type="primary" @click="save_model_confirm"
          >确定</el-button
        >
      </span>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { onMounted, nextTick, ref, reactive } from "vue";
import { jsPlumb } from "jsplumb";
import { ElNotification, ElMessage } from "element-plus";
import axios from "axios";
import { computed } from "vue";
import { useRouter } from "vue-router";
import uploadDatafile from "./uploadDatafile.vue";

import {
  labels_for_algorithms,
  algorithm_introduction,
  plain_introduction,
} from "./constant";

const router = useRouter();

const dialogVisible = ref(false);

const activeName = ref("first"); // 控制标签页

const models_drawer = ref(false);
const data_drawer = ref(false);
const file_list = ref([]);
// const Name = ref('xiaomao')

//控制按钮失效变量
const canStartProcess = ref(true);
// const canCompleteModeling = ref(true)
const canCompleteModeling = computed(() => {
  if (nodeList.value.length > 0 && !model_has_been_saved) {
    return false;
  } else {
    return true;
  }
});
const canCheckModel = ref(true);
const canSaveModel = ref(true);
const canShutdown = ref(true);

// { label: '添加噪声', id: '1.1', use_algorithm: null, parameters: { 'WhiteGaussianNoise': { SNR: -5 }, 'Noise': { param1: 10, param2: 20 } }, tip_show: false, tip: '为输入信号添加噪声' },

// 记录当前节点位置
const recordPosition = (nodeItem, { x, y }) => {
  console.log("nodeItem,--->", nodeItem);
  console.log("x,y", x, y);
};
const menuList2 = ref([
  {
    label: "预处理算法",
    id: "1",
    options: [
      {
        label: "插值处理",
        id: "1.1",
        use_algorithm: null,
        parameters: {
          neighboring_values_interpolation: {},
          polynomial_interpolation: {},
          bicubic_interpolation: {},
          lagrange_interpolation: {},
          newton_interpolation: {},
          linear_interpolation: {},
        },
        tip_show: false,
        tip: "对输入信号进行插值",
      },
      {
        label: "特征提取",
        id: "1.2",
        use_algorithm: null,
        parameters: {
          time_domain_features: {
            均值: false,
            方差: false,
            标准差: false,
            偏度: false,
            峰度: false,
            四阶累积量: false,
            六阶累积量: false,
            最大值: false,
            最小值: false,
            中位数: false,
            峰峰值: false,
            整流平均值: false,
            均方根: false,
            方根幅值: false,
            波形因子: false,
            峰值因子: false,
            脉冲因子: false,
            裕度因子: false,
          },
          frequency_domain_features: {
            重心频率: false,
            均方频率: false,
            均方根频率: false,
            频率方差: false,
            频率标准差: false,
            谱峭度的均值: false,
            谱峭度的峰度: false,
          },
          time_frequency_domain_features: {
            均值: false,
            方差: false,
            标准差: false,
            峰度: false,
            偏度: false,
            四阶累积量: false,
            六阶累积量: false,
            最大值: false,
            最小值: false,
            中位数: false,
            峰峰值: false,
            整流平均值: false,
            均方根: false,
            方根幅值: false,
            波形因子: false,
            峰值因子: false,
            脉冲因子: false,
            裕度因子: false,
            重心频率: false,
            均方频率: false,
            均方根频率: false,
            频率方差: false,
            频率标准差: false,
            谱峭度的均值: false,
            谱峭度的峰度: false,
          },
          time_domain_features_multiple: {
            均值: false,
            方差: false,
            标准差: false,
            偏度: false,
            峰度: false,
            四阶累积量: false,
            六阶累积量: false,
            最大值: false,
            最小值: false,
            中位数: false,
            峰峰值: false,
            整流平均值: false,
            均方根: false,
            方根幅值: false,
            波形因子: false,
            峰值因子: false,
            脉冲因子: false,
            裕度因子: false,
          },
          frequency_domain_features_multiple: {
            重心频率: false,
            均方频率: false,
            均方根频率: false,
            频率方差: false,
            频率标准差: false,
            谱峭度的均值: false,
            谱峭度的峰度: false,
          },
          time_frequency_domain_features_multiple: {
            均值: false,
            方差: false,
            标准差: false,
            峰度: false,
            偏度: false,
            四阶累积量: false,
            六阶累积量: false,
            最大值: false,
            最小值: false,
            中位数: false,
            峰峰值: false,
            整流平均值: false,
            均方根: false,
            方根幅值: false,
            波形因子: false,
            峰值因子: false,
            脉冲因子: false,
            裕度因子: false,
            重心频率: false,
            均方频率: false,
            均方根频率: false,
            频率方差: false,
            频率标准差: false,
            谱峭度的均值: false,
            谱峭度的峰度: false,
          },
        },
        tip_show: false,
        tip: "手工提取输入信号的特征",
      },
      {
        label: "无量纲化",
        id: "1.5",
        use_algorithm: null,
        parameters: {
          max_min: {},
          "z-score": {},
          robust_scaler: {},
          max_abs_scaler: {},
        },
        tip_show: false,
        tip: "对输入数据进行无量纲化处理",
      },
      {
        label: "特征选择",
        id: "1.3",
        use_algorithm: null,
        parameters: {
          feature_imp: {},
          mutual_information_importance: {},
          correlation_coefficient_importance: {},
          feature_imp_multiple: {},
          mutual_information_importance_multiple: {},
          correlation_coefficient_importance_multiple: {},
        },
        tip_show: false,
        tip: "对提取到的特征进行特征选择",
      },
      {
        label: "小波变换",
        id: "1.4",
        use_algorithm: null,
        parameters: {
          wavelet_trans_denoise: {},
        },
        tip_show: false,
        tip: "对输入信号进行小波变换",
      },
    ],
    tip_show: false,
    tip: "包含添加噪声、插值以及特征提取等",
  },
  {
    label: "故障预测算法",
    id: "2",
    options: [
      {
        label: "故障诊断",
        id: "2.1",
        use_algorithm: null,
        parameters: {
          random_forest: {},
          svc: {},
          gru: {},
          lstm: {},
          random_forest_multiple: {},
          svc_multiple: {},
          gru_multiple: {},
          lstm_multiple: {},
        },
        tip_show: false,
        tip: "根据提取特征对输入信号作故障诊断",
      },
      {
        label: "趋势预测",
        id: "2.2",
        use_algorithm: null,
        parameters: {
          linear_regression: {},
          linear_regression_multiple: {},
        },
        tip_show: false,
        tip: "根据提取的信号特征对输入信号进行故障预测",
      },
    ],
  },
  {
    label: "健康评估算法",
    id: "3",
    options: [
      {
        label: "层次分析模糊综合评估",
        id: "3.1",
        use_algorithm: null,
        parameters: {
          FAHP: {},
          FAHP_multiple: {},
        },
        tip_show: false,
        tip: "将模糊综合评价法和层次分析法相结合的评价方法",
      },
    ],
  },
  // {
  //   label: '语音处理', id: '2', options: [{ label: '音频分离', id: '2.1', use_algorithm: null, parameters: { 'conformer': { num_workers: 8, layers: 64 }, 'sepformer': { num_workers: 16, layers: 64 } }, tip_show: false, tip: '可对输入的一维音频信号进行噪声分离' },
  //   { label: '声纹识别', id: '2.2', use_algorithm: null, parameters: { 'conformer': {}, 'lightweight_cnn_conformer': {} }, tip_show: false, tip: '根据输入的说话人语音识别说话人' }]
  // },
]);

const background_IMG = () => {
  if (nodeList.value.length == 0) {
    document.querySelector(".el-main").classList.add("has-background");
  }
  if (nodeList.value.length >= 1) {
    document.querySelector(".el-main").classList.remove("has-background");
    document.querySelector(".el-main").style.backgroundImage = "";
  }
};

// 算法推荐参数
const recommended_settings = {
  time_domain_features: {
    配置一: {
      均值: true,
      方差: true,
      标准差: true,
      偏度: true,
      峰度: true,
      四阶累积量: true,
      六阶累积量: true,
      最大值: true,
      最小值: true,
      中位数: true,
      峰峰值: true,
      整流平均值: true,
      均方根: true,
      方根幅值: true,
      波形因子: true,
      峰值因子: true,
      脉冲因子: true,
      裕度因子: true,
    },
    config_2: {},
  },
};

// 各个算法包含的参数对应的中文名
const labels_for_params = {
  SNR: "信噪比",
  layers: "网络层数",
  num_workers: "工作线程数",
};

const introduction_to_show = ref("# 你好世界"); // 需要展示在可视化建模区的算法介绍
const showPlainIntroduction = ref(false);

// 点击标签页切换单传感器和多传感器算法
const handleClick = (tab, event) => {
  console.log(tab, event);
};

// 算法介绍，点击算法选择区内的具体算法，将其算法介绍展示在可视化建模区
const showIntroduction = (algorithm) => {
  results_view_clear();
  showStatusMessage.value = false;
  showPlainIntroduction.value = true;
  introduction_to_show.value = plain_introduction[algorithm];
};

// 算法选择菜单下拉展示
const menu_details_second = ref({});

const menu_details_third = ref({});

// 特征提取所选择的特征
// const features = ref([])
const features = ref([
  "均值",
  "方差",
  "标准差",
  "峰度",
  "偏度",
  "四阶累积量",
  "六阶累积量",
  "最大值",
  "最小值",
  "中位数",
  "峰峰值",
  "整流平均值",
  "均方根",
  "方根幅值",
  "波形因子",
  "峰值因子",
  "脉冲因子",
  "裕度因子",
  "重心频率",
  "均方频率",
  "均方根频率",
  "频率方差",
  "频率标准差",
  "谱峭度的均值",
  "谱峭度的峰度",
]);

//双向链表用于存储调用的模块顺序
class ListNode {
  constructor(value) {
    this.value = value;
    this.next = null;
  }
}

class LinkedList {
  constructor() {
    this.head = null;
    this.tail = null;
  }

  // 添加新元素到链表尾部
  append(value) {
    const newNode = new ListNode(value);

    if (!this.head) {
      this.head = newNode;
      this.tail = newNode;
    } else {
      this.tail.next = newNode;
      this.tail = newNode;
    }
  }

  // 在链表的头部添加新节点
  insertAtHead(value) {
    const newNode = new ListNode(value);

    if (!this.head) {
      this.head = newNode;
      this.tail = newNode;
    } else {
      newNode.next = this.head;
      this.head = newNode;
    }
  }

  // 打印链表所有元素
  print() {
    let current = this.head;
    while (current) {
      console.log(current.value);
      current = current.next;
    }
  }
  get_all_nodes() {
    let current = this.head;
    let nodeList = [];
    while (current) {
      nodeList.push(current.value);
      current = current.next;
    }
    return nodeList;
  }
  length() {
    if (this.head) {
      let len = 1;
      let p = this.head.next;
      while (p) {
        p = p.next;
        len += 1;
      }
      return len;
    }
    return 0;
  }
  search(target_value) {
    if (this.head == null) {
      return false;
    } else {
      let current = this.head;
      while (current) {
        if (current.value == target_value) {
          return current;
        }
        current = current.next;
      }
      return false;
    }
  }
}

const logout = () => {
  router.push("/");
  console.log("用户退出登录");
};

// 标签与节点id的转换
const display_label_to_id = (display_label) => {
  nodeList.value.forEach((node) => {
    if (node.display_label == display_label) {
      return node.id;
    }
  });
};
// const label_to_id = (label) => {
//   let node_list = nodeList.value.slice()
//   node_list.forEach(node => {
//     console.log('node: ', node)
//     if (node.label == label){
//       console.log('node_id: ', node.id)
//       return node.id
//     }
//   })
// }
function label_to_id(label) {
  let node_list = nodeList.value.slice();
  let nodeId_to_find = 0;
  node_list.forEach((node) => {
    if (node.label == label) {
      nodeId_to_find = node.id;
    }
  });
  return nodeId_to_find;
}

// 连线操作
const linkedList = new LinkedList();
onMounted(() => {
  username.value = window.localStorage.getItem("username") || "用户名未设置";
  console.log("username: ", username.value);

  document.querySelector(".el-main").classList.add("has-background");
  plumbIns = jsPlumb.getInstance();
  jsPlumbInit();

  plumbIns.bind("connection", function (info) {
    let sourceId = info.connection.sourceId;
    let targetId = info.connection.targetId;

    let id_to_label = {};

    nodeList.value.forEach((node) => {
      let id = node.id;
      let label = node.label;
      id_to_label[id] = label;
    });
    if (linkedList.head == null) {
      linkedList.append(id_to_label[sourceId]);
      linkedList.append(id_to_label[targetId]);
    } else {
      if (linkedList.head.value == id_to_label[targetId]) {
        linkedList.insertAtHead(id_to_label[sourceId]);
      } else {
        linkedList.append(id_to_label[targetId]);
      }
    }

    // 除去在linkedList中的节点，其他节点不能作为连线操作的出发点
    let linked = linkedList.get_all_nodes();
    // for(let [value, key] of id_to_label){
    //   if (linked.indexOf(key) == -1){
    //     plumbIns
    //   }
    // }

    console.log("linked: " + linked);
  });

  plumbIns.bind("beforeConnect", function (info) {
    console.log("调用");
    let sourceId = info.connection.sourceId;
    let targetId = info.connection.targetId;
    if (sourceId == "3.1") {
      return false;
    }
  });
});

const deff = {
  jsplumbSetting: {
    // 动态锚点、位置自适应
    Anchors: ["Right", "Left"],
    anchor: ["Right", "Left"],
    // 容器ID
    Container: "efContainer",
    // 连线的样式，直线或者曲线等，可选值:  StateMachine、Flowchart，Bezier、Straight
    // Connector: ['Bezier', {curviness: 100}],
    // Connector: ['Straight', { stub: 20, gap: 1 }],
    Connector: [
      "Flowchart",
      { stub: 30, gap: 1, alwaysRespectStubs: false, midpoint: 0.5, cornerRadius: 10 },
    ],
    // Connector: ['StateMachine', {margin: 5, curviness: 10, proximityLimit: 80}],
    // 鼠标不能拖动删除线
    ConnectionsDetachable: false,
    // 删除线的时候节点不删除
    DeleteEndpointsOnDetach: false,
    /**
     * 连线的两端端点类型：圆形
     * radius: 圆的半径，越大圆越大
     */
    Endpoint: ["Dot", { radius: 10, cssClass: "ef-dot", hoverClass: "ef-dot-hover" }],
    /**
     * 连线的两端端点类型：矩形
     * height: 矩形的高
     * width: 矩形的宽
     */
    // Endpoint: ['Rectangle', {height: 20, width: 20, cssClass: 'mycssClass', hoverClass: 'ef-rectangle-hover'},],
    /**
     * 图像端点
     */
    // Endpoint: ['Image', {src: 'https://www.easyicon.net/api/resizeApi.php?id=1181776&size=32', cssClass: 'ef-img', hoverClass: 'ef-img-hover'}],
    /**
     * 空白端点
     */
    // Endpoint: ['Blank', { Overlays: '' }],

    // Endpoints: [['Dot', {radius: 5, cssClass: 'ef-dot', hoverClass: 'ef-dot-hover'}], ['Rectangle', {height: 20, width: 20, cssClass: 'ef-rectangle', hoverClass: 'ef-rectangle-hover'}]],
    /**
     * 连线的两端端点样式
     * fill: 颜色值，如：#12aabb，为空不显示
     * outlineWidth: 外边线宽度
     */
    EndpointStyle: { fill: "#1879ffa1", outlineWidth: 3 },
    // 是否打开jsPlumb的内部日志记录
    LogEnabled: true,
    /**
     * 连线的样式
     */
    PaintStyle: {
      // 线的颜色
      stroke: "#4CAF50",
      // 线的粗细，值越大线越粗
      strokeWidth: 7,
      // 设置外边线的颜色，默认设置透明，这样别人就看不见了，点击线的时候可以不用精确点击，参考 https://blog.csdn.net/roymno2/article/details/72717101
      outlineStroke: "transparent",
      // 线外边的宽，值越大，线的点击范围越大
      outlineWidth: 5,
    },
    DragOptions: { cursor: "pointer", zIndex: 2000 },
    ConnectionOverlays: [
      [
        "Custom",
        {
          create() {
            const el = document.createElement("div");
            // el.innerHTML = '<select id=\'myDropDown\'><option value=\'foo\'>foo</option><option value=\'bar\'>bar</option></select>'
            return el;
          },
          location: 0.7,
          id: "customOverlay",
        },
      ],
    ],
    /**
     *  叠加 参考： https://www.jianshu.com/p/d9e9918fd928
     */
    Overlays: [
      // 箭头叠加
      [
        "Arrow",
        {
          width: 25, // 箭头尾部的宽度
          length: 8, // 从箭头的尾部到头部的距离
          location: 1, // 位置，建议使用0～1之间
          direction: 1, // 方向，默认值为1（表示向前），可选-1（表示向后）
          foldback: 0.623, // 折回，也就是尾翼的角度，默认0.623，当为1时，为正三角
        },
      ],
      // ['Diamond', {        //     events: {        //         dblclick: function (diamondOverlay, originalEvent) {        //             console.log('double click on diamond overlay for : ' + diamondOverlay.component)        //         }        //     }        // }],
      ["Label", { label: "", location: 0.1, cssClass: "aLabel" }],
    ],
    // 绘制图的模式 svg、canvas
    RenderMode: "canvas",
    // 鼠标滑过线的样式
    HoverPaintStyle: { stroke: "red", strokeWidth: 10 },
    // 滑过锚点效果
    EndpointHoverStyle: { fill: "red" },
    Scope: "jsPlumb_DefaultScope", // 范围，具有相同scope的点才可连接
  },
  /**
   * 连线参数
   */
  jsplumbConnectOptions: {
    isSource: true,
    isTarget: true,
    // 动态锚点、提供了4个方向 Continuous、AutoDefault
    // anchor: 'Continuous',
    // anchor: ['Continuous', { faces: ['left', 'right'] }],
    // 设置连线上面的label样式
    labelStyle: {
      cssClass: "flowLabel",
    },
  },
  /**
   * 源点配置参数
   */
  jsplumbSourceOptions: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: ".node-drag",
    filterExclude: false,
    anchor: ["Continuous", { faces: ["right"] }],
    // 是否允许自己连接自己
    allowLoopback: false,
    maxConnections: -1,
  },
  // 参考 https://www.cnblogs.com/mq0036/p/7942139.html
  jsplumbSourceOptions2: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: ".node-drag",
    filterExclude: false,
    // anchor: 'Continuous',
    // 是否允许自己连接自己
    allowLoopback: true,
    connector: ["Flowchart", { curviness: 50 }],
    connectorStyle: {
      // 线的颜色
      stroke: "red",
      // 线的粗细，值越大线越粗
      strokeWidth: 1,
      // 设置外边线的颜色，默认设置透明，这样别人就看不见了，点击线的时候可以不用精确点击，参考 https://blog.csdn.net/roymno2/article/details/72717101
      outlineStroke: "transparent",
      // 线外边的宽，值越大，线的点击范围越大
      outlineWidth: 10,
    },
    connectorHoverStyle: { stroke: "red", strokeWidth: 2 },
  },
  jsplumbTargetOptions: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: ".node-drag",
    filterExclude: false,
    // 是否允许自己连接自己
    anchor: ["Continuous", { faces: ["left"] }],
    allowLoopback: false,
    dropOptions: { hoverClass: "ef-drop-hover" },
  },
};
const done = ref(false);
const dialogmodle = ref(false);
//刷新
// const start = () => {
//   isshow.value = true
//   const loading = ElLoading.service({
//     lock: true,
//     text: 'Loading',
//     background: 'rgba(0, 0, 0, 0.7)',
//   })
//   setTimeout(() => {
//     loading.close()
//   }, 6000)

//   axios.request({
//     method: 'GET',
//     url: 'http://127.0.0.1:8000/homepage/',
//   });
//   setTimeout(function () {
//     // 为 iframe 的 src 属性添加一个查询参数，比如当前的时间戳，以强制刷新
//     var iframe = document.getElementById('my_gradio_app');
//     var currentSrc = iframe.src;
//     var newSrc = currentSrc.split('?')[0]; // 移除旧的查询参数
//     iframe.src = newSrc + '?updated=' + new Date().getTime();
//   }, 2000);

// }

// 刷新按钮
// const display_module = { label: '' }
// const renovate = () => {
//   if (display_module.label != '')
//     resultShow(display_module)
// }

//检查文件类型
const checkFileType = (file) => {
  console.log(file);
  console.log(file.name);
};

let model_check_right = false;
// 检查模型
const check_model = () => {
  console.log(linkedList.get_all_nodes());
  let id_to_module = {};
  let algorithms = [];
  let algorithm_schedule = [];
  let module_schedule = [];
  if (nodeList.value.length == 1) {
    module_schedule.push(nodeList.value[0].label);
    algorithm_schedule.push(nodeList.value[0].label_display);
  } else {
    module_schedule = linkedList.get_all_nodes();

    for (let i = 0; i < module_schedule.length; i++) {
      let module = module_schedule[i];
      nodeList.value.forEach((node) => {
        if (node.label == module) {
          algorithm_schedule.push(node.label_display);
        }
      });
    }
  }
  nodeList.value.forEach((node) => {
    let id = node.id;
    let label = node.label;
    let algorithm = node.label_display;
    id_to_module[id] = label;
    algorithms.push(algorithm);
  });

  let module_str = Object.values(module_schedule).join("");
  let algorithm_str = Object.values(algorithm_schedule).join("");
  console.log("module_str: " + module_str);
  console.log("algorithm_str: " + algorithm_str);
  // 判断子串后是否有更多的文本
  const moreText = (text, substring) => {
    const position = text.indexOf(substring);
    if (position === -1) {
      return false;
    }
    const endPosition = position + substring.length;
    return endPosition < text.length;
  };

  const checkSubstrings = (str, subStr1, subStrs2) => {
    const index1 = str.indexOf(subStr1);
    if (index1 !== -1) {
      // 如果 subStr1 存在
      for (const subStr2 of subStrs2) {
        const index2 = str.indexOf(subStr2, index1 + subStr1.length);
        if (index2 !== -1) {
          // 如果在 subStr1 之后找到了 subStr2 中的任何一个
          return true;
        }
      }
    }
    return false;
  };
  if (nodeList.value.length) {
    if (nodeList.value.length == 1) {
      if (
        !module_str.match("插值处理") &&
        !module_str.match("特征提取") &&
        !algorithm_str.match("GRU的故障诊断") &&
        !algorithm_str.match("LSTM的故障诊断") &&
        !algorithm_str.match("小波变换降噪") &&
        !module_str.match("无量纲化")
      ) {
        ElMessage({
          message: "该算法无法单独使用，请结合相应的算法",
          type: "warning",
        });
        return;
      } else {
        // 进行模型参数设置的检查
        let check_params_right = check_model_params();
        if (check_params_right) {
          ElMessage({
            showClose: true,
            message: "模型正常，可以保存并运行",
            type: "success",
          });
          model_check_right = true;
          updateStatus("模型建立并已通过模型检查");
        } else {
          ElMessage({
            showClose: true,
            message: "模型参数未设置",
            type: "warning",
          });
          return;
        }
      }
    } else {
      if (linkedList.length() != nodeList.value.length) {
        ElMessage({
          message: "请确保图中所有模块均已建立连接，且没有多余的模块",
          type: "warning",
        });
        return;
      } else {
        if (
          module_str.match("特征选择故障诊断") &&
          !module_str.match("特征提取特征选择故障诊断")
        ) {
          ElMessage({
            showClose: true,
            message: "因模型中包含故障诊断，建议在特征选择之前包含特征提取",
            type: "warning",
          });
          return;
        } else if (module_str.match("特征提取故障诊断")) {
          let source_id = label_to_id("特征提取");
          let current = linkedList.search("特征提取");
          let next = current.next.value;
          let target_id = label_to_id(next);
          console.log("source_id: ", source_id);
          console.log("target_id", target_id);
          let connection = plumbIns.getConnections({
            source: source_id,
            traget: target_id,
          });
          console.log("connection: ", connection);

          plumbIns.select({ source: source_id, target: target_id }).setPaintStyle({
            stroke: "#E53935",
            strokeWidth: 7,
            outlineStroke: "transparent",
            outlineWidth: 5,
          });
          ElMessage({
            showClose: true,
            message: "因模型中包含故障诊断，建议在特征提取之后包含特征选择",
            type: "warning",
          });
          return;
        } else if (
          module_str.match("层次分析模糊综合评估") &&
          !module_str.match("特征提取")
        ) {
          ElMessage({
            showClose: true,
            message: "因模型中包含层次分析模糊综合评估，建议在此之前包含特征提取",
            type: "warning",
          });
          return;
        } else if (
          module_str.match("层次分析模糊综合评估") &&
          (module_str.match("LSTM的故障诊断") || module_str.match("SVM的故障诊断"))
        ) {
          ElMessage({
            showClose: true,
            message:
              "使用深度学习模型的故障诊断无法为健康评估提供有效的评估依据，建议使用机器学习的故障诊断配合健康评估！",
            type: "warning",
          });
          return;
        } else if (
          module_str.match("层次分析模糊综合评估") &&
          moreText(module_str, "层次分析模糊综合评估")
        ) {
          ElMessage({
            showClose: true,
            message: "注意健康评估之后无法连接更多的模块",
            type: "warning",
          });
          return;
        } else if (algorithm_str.match("多传感器") && algorithm_str.match("单传感器")) {
          ElMessage({
            showClose: true,
            message: "针对单传感器的算法无法与针对多传感器的算法共用",
            type: "warning",
          });
          return;
        } else if (module_str.match("故障诊断")) {
          if (moreText(module_str, "故障诊断")) {
            if (
              !checkSubstrings(module_str, "故障诊断", [
                "层次分析模糊综合评估",
                "趋势预测",
              ])
            ) {
              ElMessage({
                showClose: true,
                message: "注意故障诊断之后仅能进行趋势预测或是健康评估！",
                type: "warning",
              });
              let source_id = label_to_id("故障诊断");
              let current = linkedList.search("故障诊断");
              let next = current.next.value;
              let target_id = label_to_id(next);
              console.log("source_id: ", source_id);
              console.log("target_id", target_id);
              let connection = plumbIns.getConnections({
                source: source_id,
                traget: target_id,
              });
              console.log("connection: ", connection);

              plumbIns.select({ source: source_id, target: target_id }).setPaintStyle({
                stroke: "#E53935",
                strokeWidth: 7,
                outlineStroke: "transparent",
                outlineWidth: 5,
              });
              // connection.addOverlay(
              //   [
              //     "Custom", {
              //       create: function(component) {
              //         // 创建一个img元素来引用SVG文件（注意：这里使用相对路径或Vue CLI处理后的路径）
              //         var img = document.createElement('img');
              //         img.src = require('/assets/叉号.svg'); // 注意：这里的@是Vue CLI的别名，指向src目录
              //         img.style.width = '20px'; // 设置图标大小
              //         img.style.height = 'auto'; // 保持宽高比

              //         // 或者，如果你想要直接嵌入SVG代码（假设你已经有了SVG的字符串表示）
              //         // var svgString = '<svg>...</svg>'; // SVG代码
              //         // var parser = new DOMParser();
              //         // var svgDoc = parser.parseFromString(svgString, "image/svg+xml");
              //         // var svgElement = svgDoc.documentElement;
              //         // return svgElement; // 直接返回SVG元素

              //         return img; // 返回img元素作为overlay
              //       },
              //       location: 0.5, // 在连接的中点添加overlay
              //       id: "crossOverlay" // 可选ID
              //     }
              //   ]
              // )

              // console.log('connection2: ', connection2)
              return;
            }
          }
          if (algorithm_str.match("SVM的故障诊断")) {
            if (
              !module_str.match("无量纲化") ||
              !checkSubstrings(module_str, "无量纲化", ["故障诊断"])
            ) {
              ElMessage({
                showClose: true,
                message: "因模型中包含SVM的故障诊断，需要在此之前加入标准化操作",
                type: "warning",
              });
              return;
            }
          }
        }
        // else if (algorithm_str.match('SVM的故障诊断')) {
        //   if (!module_str.match('无量纲化')){
        //     ElMessage({
        //       showClose: true,
        //       message: '因模型中包含SVM的故障诊断，需要在此之前加入标准化操作',
        //       type: 'warning'
        //     })
        //     return
        //   }
        // }
        // 进行模型参数设置的检查
        let check_params_right = check_model_params();
        if (check_params_right) {
          ElMessage({
            showClose: true,
            message: "模型正常，可以保存并运行",
            type: "success",
          });
          model_check_right = true;
          updateStatus("模型建立并已通过模型检查");
        } else {
          ElMessage({
            showClose: true,
            message: "模型参数未设置",
            type: "warning",
          });
          return;
        }
      }
    }
  } else {
    ElMessage({
      message: "请先建立模型",
      type: "warning",
    });
    return;
  }
  canSaveModel.value = false;
  // canStartProcess.value = false
};

// 进度条完成度
let processing = ref(false);
let percentage = ref(0);
// let timerId = null
let fastTimerId = null; // 快速定时器ID
let slowTimerId = null; // 慢速定时器ID

let response_results = {};

const username = ref("");

//上传文件，点击开始
const run = () => {
  // console.log("文件列表:", Object.keys(content_json.algorithms).length)
  // console.log("json:", content_json)
  // 发送文件到后端
  if (!file_list.value.length) {
    ElNotification({
      title: "WARNING",
      message: "数据文件不能为空,请先上传数据",
      type: "warning",
    });
    return;
  } else {
    const filename = file_list.value[0].name;
    const filetype = filename.substring(filename.lastIndexOf("."));
    if (
      filetype != ".xlsx" &&
      filetype != ".npy" &&
      filetype != ".wav" &&
      filetype != ".audio" &&
      filetype != ".csv" &&
      filetype != ".mat"
    ) {
      ElNotification({
        title: "WARNING",
        message: "文件类型只能为xlsx、npy、wav、audio、csv、mat",
        type: "warning",
      });
      return;
    }
  }

  let datafile = file_list.value[0].raw;

  const data = new FormData();
  data.append("file", datafile);
  data.append("params", JSON.stringify(content_json));
  ElNotification.info({
    title: "SUCCESS",
    message: "正在处理，请等待",
  });
  canShutdown.value = false;

  percentage.value = 0; // 重置进度条

  // 启动定时器来模拟进度条更新
  // timerId = setInterval(() => {
  //   if (percentage.value < 90) {
  //     percentage.value += 10;
  //   }
  //   // 如果达到90%，则不再增加，但定时器仍然运行以等待后端响应
  // }, 1000);
  // 启动快速定时器
  fastTimerId = setInterval(() => {
    if (percentage.value < 50) {
      percentage.value += 10;
    } else {
      // 达到50%时，清除快速定时器并启动慢速定时器
      clearInterval(fastTimerId);
      slowTimerId = setInterval(() => {
        if (percentage.value < 90) {
          percentage.value += 10;
        } else {
          // 达到100%时清除慢速定时器
          clearInterval(slowTimerId);
        }
      }, 3000); // 每三秒更新一次
    }
  }, 1000); // 每秒更新一次（在进度小于50%时）
  processing.value = true;

  // console.log("",data)
  api
    .post("http://127.0.0.1:8000/homepage/", data, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((response) => {
      console.log("response: ", response);
      console.log("response.status: ", response.status);
      if (response.status === 200) {
        if (!process_is_shutdown) {
          ElNotification.success({
            title: "SUCCESS",
            message: "处理完成",
          });
          response_results = response.data.results;
          console.log("resoonse_results: ", response_results);
          missionComplete.value = true;
          // setTimeout(function () { processing.value = false; percentage.value = 50 }, 500)
          // percentage.value = 100
          // clearInterval(timerId);
          clearInterval(fastTimerId);
          clearInterval(slowTimerId);
          setTimeout(function () {
            processing.value = false;
          }, 700);
          percentage.value = 100;
          canShutdown.value = true;
          status_message_to_show.value = status_message.success;
          showStatusMessage.value = true;
          showPlainIntroduction.value = false;
        } else {
          process_is_shutdown = false;
        }
      }
      // else if (response.status === 500) {
      //   ElNotification.error({
      //     title: 'ERROR',
      //     message: '处理失败，请重试',
      //   })
      //   loading.value = false
      // }
    })
    .catch((error) => {
      if (error.response) {
        // 请求已发出，服务器响应了状态码，但不在2xx范围内
        console.log(error.response.status); // HTTP状态码
        console.log(error.response.statusText); // 状态消息
      } else if (error.request) {
        // 请求已发起，但没有收到响应
        console.log(error.request);
      } else {
        // 设置请求时触发了错误
        console.error("Error", error.message);
      }

      ElNotification.error({
        title: "ERROR",
        message: "处理失败，请重试",
      });
      loading.value = false;
      processing.value = false;
    });
};

let process_is_shutdown = false;

// 终止进程
const shutdown = () => {
  axios
    .request({
      method: "GET",
      url: "http://127.0.0.1:8000/shut",
    })
    .then((response) => {
      if (response.data.status == "shutdown" && processing.value == true) {
        loading.value = false;
        processing.value = false;
        missionComplete.value = false;
        ElNotification.info({
          title: "INFO",
          message: "进程已终止",
        });
        process_is_shutdown = true;
        status_message_to_show.value = status_message.shutdown;
        showStatusMessage.value = true;
      }
    })
    .catch(function (error) {
      // 处理错误情况
      ElNotification.error({
        title: "ERROR",
        message: "请求终端进程失败",
      });
      console.log("请求中断进程失败：" + error);
    });
};

const isshow = ref(false);
const selects = ref(false);
const nodeList = ref([]);
const efContainerRef = ref();
const nodeRef = ref([]);
const content_json = {
  modules: [],
  algorithms: {},
  parameters: {},
  schedule: [],
};

let plumbIns;
let missionComplete = ref(false);
let loading = ref(false);
let modelSetup = ref(false);
const handleclear = () => {
  done.value = false;
  nodeList.value = []; // 可视化建模区的节点列表
  // features.value = []  // 特征提取选择的特征
  features.value = [
    "均值",
    "方差",
    "标准差",
    "峰度",
    "偏度",
    "四阶累积量",
    "六阶累积量",
    "最大值",
    "最小值",
    "中位数",
    "峰峰值",
    "整流平均值",
    "均方根",
    "方根幅值",
    "波形因子",
    "峰值因子",
    "脉冲因子",
    "裕度因子",
    "重心频率",
    "均方频率",
    "均方根频率",
    "频率方差",
    "频率标准差",
    "谱峭度的均值",
    "谱峭度的标准差",
    "谱峭度的峰度",
    "谱峭度的偏度",
  ];
  json_file_clear(); // 向后端发送的模型信息
  isshow.value = false;
  plumbIns.deleteEveryConnection();
  plumbIns.deleteEveryEndpoint();
  linkedList.head = null;
  linkedList.tail = null;
  missionComplete.value = false; // 程序处理完成
  modelSetup.value = false; // 模型设置完成
  showPlainIntroduction.value = false;
  showStatusMessage.value = false;
  model_has_been_saved = false; //复用历史模型，不做模型检查
  to_rectify_model.value = false; // 禁用修改模型
  canCompleteModeling.value = true;
  canCheckModel.value = true;
  canSaveModel.value = true;
  canStartProcess.value = true;
  updateStatus("未建立模型");

  results_view_clear();
};
const json_file_clear = () => {
  content_json.modules = [];
  content_json.algorithms = {};
  content_json.parameters = {};
  content_json.schedule = [];
};
const jsPlumbInit = () => {
  plumbIns.importDefaults(deff.jsplumbSetting);
};

//处理拖拽，初始化节点的可连接状态
const handleDragend = ({ clientX, clientY }, algorithm, node) => {
  console.log("拖拽结束", clientX, clientY);
  console.log("node: ", node);

  let left = clientX - 300 + "px";
  let top = 50 + "px";
  const nodeId = node.id;
  const nodeInfo = {
    label_display: labels_for_algorithms[algorithm],
    label: node.label,
    id: node.id,
    nodeId,
    nodeContainerStyle: {
      left: left,
      top: top,
    },
    use_algorithm: algorithm,
    parameters: node.parameters,
  };
  console.log(nodeInfo);
  //算法模块不允许重复
  if (nodeList.value.length === 0) {
    nodeList.value.push(nodeInfo);
  } else {
    let isDuplicate = false;
    for (let i = 0; i < nodeList.value.length; i++) {
      let nod = nodeList.value[i];
      if (nod.id == node.id) {
        // window.alert('不允许出现重复模块');
        ElMessage({
          message: "不允许出现同一类别的算法",
          type: "warning",
        });
        isDuplicate = true;
        break;
      }
    }
    if (!isDuplicate) {
      nodeList.value.push(nodeInfo);
    }
  }
  // nodeList.value.push(nodeInfo)
  // 将节点初始化为可以连线的状态
  nextTick(() => {
    plumbIns.draggable(nodeId, { containment: "efContainer" });
    // if (node.id > 2) {
    //   plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)
    //   return
    // }
    if (node.id < 4) {
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions);
    }

    // plumbIns.addEndpoint(nodeId, deff.jsplumbTargetOptions)

    plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions);
  });
};

// 删除节点的操作
const deleteNode = (nodeId) => {
  if (!modelSetup.value) {
    nodeList.value = nodeList.value.filter((node) => node.id !== nodeId);
    plumbIns.deleteEveryConnection();
    plumbIns.deleteEveryEndpoint();
    linkedList.head = null;
    linkedList.tail = null;
    canCheckModel.value = true;
    canStartProcess.value = true;
    canShutdown.value = true;
    canSaveModel.value = true;
  }
};

const modelsetting = () => {
  selects.value = !selects.value;
};

const dialogFormVisible = ref(false); // 控制对话框弹出，输入要保存的模型的名称

// 提交的模型相关信息
const model_info_form = ref({
  name: "",
  region: "",
});

// 检查模型参数设置
const check_model_params = () => {
  for (let i = 0; i < nodeList.value.length; i++) {
    let dict = nodeList.value[i];

    if (!dict.use_algorithm) {
      return false;
    }

    if (dict.id == "1.2") {
      if (!features.value.length) {
        return false;
      }
    }
  }

  return true;
};

//保存模型并取消拖拽动作
const saveModelSetting = (saveModel, schedule) => {
  done.value = true;

  // dialogFormVisible.value = true
  // selects.value = !selects.value
  json_file_clear();
  for (let i = 0; i < nodeList.value.length; i++) {
    let dict = nodeList.value[i];

    if (!dict.use_algorithm) {
      ElMessage({
        message: "请设置每个算法的必选属性",
        type: "error",
      });
      console.log("dict.use_algorithm is empty! return");
      return;
    }

    content_json.algorithms[dict.label] = dict.use_algorithm;
    if (!content_json.modules.includes(dict.label)) {
      content_json.modules.push(dict.label);
    }

    // 选择特征提取需要展示的参数
    if (dict.id == "1.2") {
      let params = dict.parameters[dict.use_algorithm];
      if (!features.value.length) {
        ElMessage({
          message: "请设置每个算法的必选属性",
          type: "error",
        });
        return;
      }
      features.value.forEach((element) => {
        if (params[element] == false) {
          params[element] = true;
        }
      });
      content_json.parameters[dict.use_algorithm] = params;
      continue;
    }
    content_json.parameters[dict.use_algorithm] = dict.parameters[dict.use_algorithm];
    // console.log(dict.use_algorithm + '\'s params are: ' + dict.parameters[dict.use_algorithm])
  }
  if (!model_check_right && saveModel) {
    ElMessage({
      message: "请先建立模型并通过模型检查！",
      type: "warning",
    });
    return;
  }
  let current = linkedList.head;
  content_json.schedule.length = 0;
  console.log("nodeList: ", nodeList.value);
  // 如果只有一个节点，无需建立流程
  if (nodeList.value.length == 1) {
    content_json.schedule.push(nodeList.value[0].label);
    // console.log('只有一个节点，无需建立流程')
  } else {
    if (!saveModel) {
      console.log("schedule: ", schedule);
      content_json.schedule = schedule;
      console.log("content_json: ", content_json);
    } else {
      if (!current) {
        ElNotification({
          title: "WARNING",
          message: "未建立流程，请先建立流程",
          type: "warning",
        });
        return;
      }
    }
    while (current) {
      content_json.schedule.push(current.value);
      current = current.next;
    }
  }

  // for (let i = 0; i < nodeList.value.length; i++) {
  //   let nod = nodeList.value[i];
  //   plumbIns.setDraggable(nod.nodeId)
  // }
  // modelSetup = true
  // dialogFormVisible.value = saveModel   // 弹出模型保存相关信息的对话框
  // dialogFormVisible.value =true
  dialogmodle.value = saveModel;
};

// 完成模型名称等信息的填写后，确定保存模型
const save_model_confirm = () => {
  let data = new FormData();
  data.append("model_name", model_info_form.value.name);
  let nodelist_length = nodeList.value.length;
  let nodelist_info = nodeList.value;
  // for (let i = 0; i < nodelist_length; i++){
  //   nodelist_info[i].nodeContainerStyle.left += 'px'
  //   nodelist_info[i].nodeContainerStyle.top += 'px'
  // }

  let model_info = { nodeList: nodelist_info, connection: content_json.schedule };
  data.append("model_info", JSON.stringify(model_info));
  // data.append('username', window.localStorage.getItem('username'))
  // axios.post('http://127.0.0.1:8000/save_model/', data,
  //   {
  //     headers: { "Content-Type": 'multipart/form-data' }
  //   }
  // ).then((response) => {
  //   if (response.data.message == 'save_model_successful') {
  //     ElMessage({
  //       message: '保存模型成功',
  //       type: 'success'
  //     })
  //     fetch_models()
  //     models_drawer.value = false       // 关闭历史模型抽屉
  //     dialogFormVisible.value = false    // 关闭提示窗口
  //     dialogmodle.value = false
  //     canStartProcess.value = false     // 保存模型成功可以运行
  //     modelSetup.value = true                 // 模型保存完成
  //     updateStatus('当前模型已保存')
  //   } else {
  //     ElMessage({
  //       message: '保存模型失败',
  //       type: 'error'
  //     })
  //   }
  // })
  api
    .post("http://127.0.0.1:8000/user_save_model/", data, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((response) => {
      if (response.data.message == "save model success") {
        ElMessage({
          message: "保存模型成功",
          type: "success",
        });
        fetch_models();
        models_drawer.value = false; // 关闭历史模型抽屉
        dialogFormVisible.value = false; // 关闭提示窗口
        dialogmodle.value = false;
        canStartProcess.value = false; // 保存模型成功可以运行
        modelSetup.value = true; // 模型保存完成
        updateStatus("当前模型已保存");
      } else if (response.data.status == 400) {
        ElMessage({
          message: "已有同名模型，保存模型失败",
          type: "error",
        });
      }
    })
    .catch((error) => {
      ElMessage({
        message: "保存模型请求失败",
        type: "error",
      });
      console.log("save model error: ", error);
    });
};
// let showMenu = ref(false)
// const handleRightClick = (event) => {
//   event.preventDefault(); // 阻止默认的右键菜单
//   showMenu.value = true; // 显示自定义的下拉菜单
//   console.log('右击')
//   console.log(event.target.getAttribute('id'))
// }
const show1 = ref(false);

// 结果可视化区域显示（无后端响应时间限制）
const result_show = ref(false);

// 健康评估结果展示
const health_evaluation = ref("");
const display_health_evaluation = ref(false);
const activeName1 = ref("first");
const health_evaluation_figure_1 = ref("data:image/png;base64,");
const health_evaluation_figure_2 = ref("data:image/png;base64,");
const health_evaluation_figure_3 = ref("data:image/png;base64,");

const health_evaluation_display = (results_object) => {
  display_health_evaluation.value = true;
  let figure1 = results_object.层级有效指标_Base64;
  let figure2 = results_object.二级指标权重柱状图_Base64;
  let figure3 = results_object.评估结果柱状图_Base64;

  health_evaluation.value = results_object.评估建议;
  health_evaluation_figure_1.value = "data:image/png;base64," + figure1;
  health_evaluation_figure_2.value = "data:image/png;base64," + figure2;
  health_evaluation_figure_3.value = "data:image/png;base64," + figure3;

  // const imgElement1 = document.getElementById('health_evaluation_figure_1');
  // const imgElement2 = document.getElementById('health_evaluation_figure_2');
  // const imgElement3 = document.getElementById('health_evaluation_figure_3');
  // imgElement1.src = `data:image/png;base64,${figure1}`;
  // imgElement2.src = `data:image/png;base64,${figure2}`;
  // imgElement3.src = `data:image/png;base64,${figure3}`;
};
const health_evaluation_hide = () => {
  display_health_evaluation.value = false;
};

// 特征提取结果展示
const display_feature_extraction = ref(false);
const transformedData = ref([]);
const columns = ref([]);

const feature_extraction_display = (results_object) => {
  // console.log('提取到的特征： ', results_object)
  display_feature_extraction.value = true;
  let features_with_name = Object.assign({}, results_object.features_with_name);
  let features_name = features_with_name.features_name.slice();
  let features_group_by_sensor = Object.assign(
    features_with_name.features_extracted_group_by_sensor
  );
  let datas = []; // 表格中每一行的数据
  features_name.unshift("传感器"); // 表格的列名
  for (const sensor in features_group_by_sensor) {
    let features_of_sensor = features_group_by_sensor[sensor].slice();
    features_of_sensor.unshift(sensor);
    console.log("features_of_sensor: ", features_of_sensor);
    datas.push(features_of_sensor);
    // features_of_sensor.splice(0, 1)
  }
  console.log("features_name: ", features_name);
  console.log("datas: ", datas);
  let i = { prop: "", label: "", width: "" };
  columns.value.length = 0;
  features_name.forEach((element) => {
    console.log("element: ", element);
    columns.value.push({ prop: element, label: element, width: 180 });
  });

  console.log("columns: ", columns);
  // 转换数据为对象数组
  transformedData.value = datas.map((row, index) => {
    const obj = {};
    columns.value.forEach((column, colIndex) => {
      obj[column.prop] = row[colIndex];
    });
    return obj;
  });
  console.log("transformedData: ", transformedData);
  // console.log('table_datas: ', datas)
  // console.log('table_columns: ', features_name)
  // features_name.splice(0, 1)

  // 构造el-table数据对象
};

// 特征提取结果可视化
const display_feature_selection = ref(false);
const feature_selection_figure = ref("");
const features_selected = ref("");

const features_selection_display = (results_object) => {
  display_feature_selection.value = true;

  let figure1 = results_object.figure_Base64;
  features_selected.value = results_object.selected_features.join("、");
  feature_selection_figure.value = "data:image/png;base64," + figure1;
};

// 故障诊断结果展示
const display_fault_diagnosis = ref(false);
const fault_diagnosis = ref("");
const fault_diagnosis_figure = ref("");

const fault_diagnosis_display = (results_object) => {
  display_fault_diagnosis.value = true;

  let figure1 = results_object.figure_Base64;
  let diagnosis_result = results_object.diagnosis_result;
  if (diagnosis_result == 0) {
    fault_diagnosis.value = "无故障";
  } else {
    fault_diagnosis.value = "存在故障";
  }
  fault_diagnosis_figure.value = "data:image/png;base64," + figure1;
};

// 故障预测结果展示
const display_fault_regression = ref(false);
const have_fault = ref(0);
const fault_regression = ref("");
const time_to_fault = ref("");
const fault_regression_figure = ref("");

const fault_regression_display = (results_object) => {
  display_fault_regression.value = true;

  let figure1 = results_object.figure_Base64;
  fault_regression_figure.value = "data:image/png;base64," + figure1;
  // let fault_time = results_object.time_to_fault

  if (results_object.time_to_fault == 0) {
    have_fault.value = 1;
    fault_regression.value = "已经出现故障";
  } else {
    have_fault.value = 0;
    fault_regression.value = "还未出现故障";
    time_to_fault.value = results_object.time_to_fault_str;
  }
};

// 插值处理可视化
const display_interpolation = ref(false);
const interpolation_figure = ref("");

const interpolation_display = (results_object) => {
  display_interpolation.value = true;

  let figure1 = results_object.figure_Base64;
  interpolation_figure.value = "data:image/png;base64," + figure1;
};

// 清除可视化区域
const results_view_clear = () => {
  showPlainIntroduction.value = false; // 清除算法介绍
  showStatusMessage.value = false; // 清除程序运行状态
  result_show.value = false; // 清除可视化区域元素
  show1.value = true;
  loading.value = true;
  isshow.value = false;
  // 清除所有结果可视化
  display_health_evaluation.value = false;
  display_feature_extraction.value = false;
  display_feature_selection.value = false;
  display_fault_diagnosis.value = false;
  display_fault_regression.value = false;
  display_interpolation.value = false;
};

// 做一个缓冲，以免用户快速点击致使后端崩溃
const refreshPardon = ref("http://127.0.0.1:7860");

// 根据节点信息发送结果展示请求
const resultShow = (item) => {
  // showMenu.value = false
  // isshow.value = false
  // isshow.value = true
  // display_module.label = item.label
  // console.log(display_module[label])
  if (done.value) {
    if (missionComplete.value) {
      if (
        item.label != "层次分析模糊综合评估" &&
        item.label != "特征提取" &&
        item.label != "特征选择" &&
        item.label != "故障诊断" &&
        item.label != "趋势预测" &&
        item.label != "特征提取"
      ) {
        showPlainIntroduction.value = false;
        showStatusMessage.value = false;
        show1.value = true;
        loading.value = true;
        result_show.value = false;
        isshow.value = false;
        setTimeout(function () {
          isshow.value = true;
          show1.value = false;
          loading.value = false;
        }, 2500); // 5000毫秒等于5秒
        let moduleName = item.label;
        let url = "http://127.0.0.1:8000/homepage?display=" + moduleName;
        axios.request({
          method: "GET",
          url: url,
        });
        setTimeout(function () {
          // 为 iframe 的 src 属性添加一个查询参数，比如当前的时间戳，以强制刷新
          var iframe = document.getElementById("my_gradio_app");
          var currentSrc = iframe.src;
          var newSrc = currentSrc.split("?")[0]; // 移除旧的查询参数
          iframe.src = newSrc + "?updated=" + new Date().getTime();
        }, 2400);
      } else {
        results_view_clear();
        result_show.value = true;
        if (item.label == "层次分析模糊综合评估") {
          let results_to_show = response_results.层次分析模糊综合评估;
          health_evaluation_display(results_to_show);
        } else if (item.label == "特征提取") {
          let results_to_show = response_results.特征提取;
          feature_extraction_display(results_to_show);
        } else if (item.label == "特征选择") {
          let results_to_show = response_results.特征选择;
          features_selection_display(results_to_show);
        } else if (item.label == "故障诊断") {
          let results_to_show = response_results.故障诊断;
          fault_diagnosis_display(results_to_show);
        } else if (item.label == "趋势预测") {
          let results_to_show = response_results.趋势预测;
          fault_regression_display(results_to_show);
        } else if (item.label == "插值处理") {
          let results_to_show = response_results.插值处理;
          interpolation_display(results_to_show);
        } else {
          ElMessage({
            message: "无效的算法模块",
            type: "error",
          });
        }
      }
    }
  } else {
  }
};

// 用户历史数据
// const fetchedDatasetsInfo = ref([])

// const fetch_dataset = () => {
//   models_drawer.value = false
//   data_drawer.value = true
//   let url = 'http://127.0.0.1:8000/fetch_datafiles/'
//   api.request({
//     method: 'GET',
//     url: url
//   })
//   .then(response => {
//     let datasetInfo = response.data
//     fetchedDatasetsInfo.value.length = 0
//     for (let item of datasetInfo){
//       fetchedDatasetsInfo.value.push(item)
//     }
//   })
// }

// 打开抽屉，同时从后端获取历史模型
const fetch_models = () => {
  data_drawer.value = false;
  models_drawer.value = true;
  let url = "http://127.0.0.1:8000/user_fetch_models/";
  api
    .request({
      method: "GET",
      url: url,
    })
    .then((response) => {
      let modelsInfo = response.data;
      fetchedModelsInfo.value.length = 0;
      for (let item of modelsInfo) {
        fetchedModelsInfo.value.push(item);
      }
    });
};

// 从后端获取到的历史模型的信息
const fetchedModelsInfo = ref([]);

// 复用历史模型，不需要进行模型检查等操作
let model_has_been_saved = false;

// 点击历史模型表格中使用按钮触发复现历史模型
const use_model = (row) => {
  if (nodeList.value.length != 0) {
    nodeList.value.length = 0;
  }
  handleclear();
  updateStatus("当前模型已保存");
  model_has_been_saved = true;
  canStartProcess.value = false;
  let objects = JSON.parse(row.model_info);
  let node_list = objects.nodeList; // 模型节点信息
  let connection = objects.connection; // 模型连线信息
  // for (let i = 0; i < node_list.length; i++){

  //   node_list[i].nodeContainerStyle.left -= 490
  //   node_list[i].nodeContainerStyle.top -= 120
  //   node_list[i].nodeContainerStyle.left += 'px'
  //   node_list[i].nodeContainerStyle.top += 'px'

  // }
  // 恢复节点
  for (let node of node_list) {
    nodeList.value.push(node);

    if (node.label == "特征提取") {
      features.value.length = 0;
      let params = node.parameters[node.use_algorithm];
      for (let [key, value] of Object.entries(params)) {
        if (value) {
          features.value.push(key);
        }
      }
    }
  }
  let id_to_label_list = { node_id: [], node_label: [] };
  // 初始化每个节点的可连接状态
  for (let node of nodeList.value) {
    let nodeId = node.id;
    id_to_label_list.node_id.push(nodeId);
    id_to_label_list.node_label.push(node.label);
    nextTick(() => {
      // plumbIns.draggable(nodeId, { containment: "efContainer" })
      if (node.id === "2.2") {
        plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions);
        return;
      }
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions);
      // plumbIns.addEndpoint(nodeId, deff.jsplumbTargetOptions)
      if (node.id === "1") {
        return;
      }
      plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions);
    });
  }

  // 恢复连线
  let connection_list = [];
  let connection2 = [];
  let node_num = connection.length;
  console.log(connection);
  for (let i = 0; i < node_num; i++) {
    let label = connection[i];
    for (let j = 0; j < node_num; j++) {
      if (id_to_label_list.node_label[j] === label) {
        connection2[i] = id_to_label_list.node_id[j];
        break;
      }
    }
  }
  console.log(connection2);
  saveModelSetting(false, connection);
  content_json.schedule = connection;
  console.log("conten_json3: ", content_json);
  modelSetup.value = true;
  if (node_num == 1) {
    connection_list = [];
  } else {
    for (let i = 0; i < node_num - 1; i++) {
      connection_list.push({ soruce_id: connection2[i], target_id: connection2[i + 1] });
    }
    nextTick(() => {
      for (let line of connection_list) {
        plumbIns.connect({
          source: document.getElementById(line.soruce_id),
          target: document.getElementById(line.target_id),
        });
      }
    });
    // for (let line of connection_list) {
    //     plumbIns.connect({
    //       source: document.getElementById(line.soruce_id),
    //       target: document.getElementById(line.target_id)
    //     })
    //   }
  }
  // jsPlumb.connect({
  //   source: div1, // 起点
  //   target: div2, // 终点
  // endpoint: "Dot", // 端点的样式，这里使用了内置的 "Dot" 样式
  // anchor: ["Right", "Left"], // 起点和终点的锚点位置
  // connector: ["Flowchart"], // 连线的样式，这里使用了内置的 "Flowchart" 样式
  // paintStyle: { stroke: "blue", strokeWidth: 2 }, // 连线的样式
  // hoverPaintStyle: { stroke: "red", strokeWidth: 4 } // 当鼠标悬停在连线上时的样式
  // });
  // console.log(nodeList.value)
  // toggleContent(false)
  // console.log(content_json)
};

// 删除模型操作
let index = 0;
let row = 0;
const delete_model_confirm_visible = ref(false);

const delete_model = (index_in, row_in) => {
  index = index_in;
  row = row_in;
  delete_model_confirm_visible.value = true;
};

const delete_model_confirm = () => {
  // 发送删除请求到后端，row 是要删除的数据行
  let url = "http://127.0.0.1:8000/user_delete_model/?row_id=" + row.id;
  api
    .request({
      method: "GET",
      url: url,
    })
    .then((response) => {
      if (response.data.message == "deleteSuccessful") {
        if (index !== -1) {
          // 删除前端表中数据
          fetchedModelsInfo.value.splice(index, 1);
          delete_model_confirm_visible.value = false;
        }
      }
    })
    .catch((error) => {
      // 处理错误，例如显示一个错误消息
      console.error(error);
    });
};

// 查看模型信息
const model_name = ref("");
const model_algorithms = ref([]);
const model_params = ref([]); // {'模块名': xx, '算法': xx, '参数': xx}

const show_model_info = (row) => {
  let objects = JSON.parse(row.model_info);
  let node_list = objects.nodeList; // 模型节点信息
  let connection = objects.connection; // 模型连接顺序

  model_name.value = row.name;
  model_algorithms.value = connection;
  model_params.value.length = 0;
  node_list.forEach((element) => {
    let item = { 模块名: "", 算法: "" };
    item.模块名 = element.label;
    item.算法 = labels_for_algorithms[element.use_algorithm];
    model_params.value.push(item);
  });
};

const showStatusMessage = ref(false);
const status_message_to_show = ref("");

const status_message = {
  success: "## 程序已经运行完毕，请点击相应的算法模块查看对应结果！",
  shutdown: "## 程序运行终止，点击重置模型重新建立模型",
};

//按钮失效控制

// 控制是否可以修改模型
const to_rectify_model = ref(false);

//控制完成建模按钮是否可以使用
// const CompleteModeling = () => {
//   if (nodeList.value.length && !model_has_been_saved) {
//     canCompleteModeling.value = false
//   } else {
//     canCompleteModeling.value = true
//   }
// }

// 完成建模动作
const finished_model = () => {
  if (nodeList.value.length) {
    if (linkedList.length() == 0 && nodeList.value.length == 1) {
      ElMessage({
        message: "完成建模",
        type: "success",
      });
      canCheckModel.value = false;
      modelSetup.value = true; // 不能删除建模区模块
      done.value = true; // 不能拖动模块
      to_rectify_model.value = true;
      updateStatus("模型建立完成但未通过检查");
      return;
    }
    if (linkedList.length() != nodeList.value.length) {
      ElMessage({
        message: "请确保图中所有模块均已建立连接，且没有多余的模块",
        type: "warning",
      });
      return;
    }
  }

  ElMessage({
    message: "完成建模",
    type: "success",
  });
  modelSetup.value = true; // 不能删除建模区模块
  done.value = true; // 不能拖动模块
  to_rectify_model.value = true;
  canCheckModel.value = false;
  updateStatus("模型建立完成但未通过检查");
};

// 修改模型
const rectify_model = () => {
  canCheckModel.value = true;
  canSaveModel.value = true;
  canStartProcess.value = true;
  canShutdown.value = true;
  modelSetup.value = false; // 可以删除建模区模块
  done.value = false; // 可以拖动模块
  to_rectify_model.value = false;
  ElMessage({
    showClose: true,
    message: "进行模型修改, 完成修改后请再次点击完成建模",
    type: "info",
  });
  updateStatus("正在修改模型");
};

//检查模型
const checkModeling = () => {
  if (nodeList.value.length == 0 && !model_has_been_saved) {
    canCheckModel.value = true;
  }
};

//保存模型
const saveModeling = () => {
  if (nodeList.value.length == 0 || model_has_been_saved) {
    canSaveModel.value = true;
  }
};

//开始
const startModeling = () => {
  if (nodeList.value.length == 0) {
    canStartProcess.value = true;
  }
};

//删除文件
const handleDelete = (file) => {
  // 使用splice方法并确保使用其返回值
  const index = file_list.value.findIndex((f) => f.uid === file.uid);
  if (index !== -1) {
    file_list.value.splice(index, 1)[0];
  }
};

// 建模状态更新
function updateStatus(status) {
  var indicator = document.getElementById("statusIndicator");
  indicator.textContent = status; // 更新文本
  indicator.classList.remove("error", "success", "saved", "rectify"); // 移除之前的状态类
  switch (status) {
    case "未建立模型":
      // 默认样式，或者设置为特定类
      break;
    case "模型建立完成但未通过检查":
      indicator.classList.add("error");
      break;
    case "模型建立并已通过模型检查":
      indicator.classList.add("success");
      break;
    case "当前模型已保存":
      indicator.classList.add("saved");
    case "正在修改模型":
      indicator.classList.add("rectify");
      break;
  }
}

// 上传文件到服务端
// const loadingDataModel = ref('local')
// const upload_data_to_server = () => {
//   let datafile = file_list.value[0].raw

//   const data = new FormData()
//   data.append("file", datafile)

//   api.post(
//     '/upload_datafile/',
//     data
//   ).then(response => {
//     if (response.data.message == 'save data success'){
//       ElMessage({
//         message: '成功上传文件到服务器',
//         type: 'success'
//       })
//     }else if (response.data.message == 'data file already exists'){
//       ElMessage({
//         message: '同名文件已经存在',
//         type: 'error'
//       })
//     }
//   })
//   .catch(error => {
//     console.log('upload_data_to_server_error: ', error)
//     ElMessage({
//       message: '上传文件失败',
//       type: 'error'
//     })
//   })
// }

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

// 拦截请求并将登录时从服务器获得的token添加到Authorization头部
api.interceptors.request.use(
  function (config) {
    // 从localStorage获取JWT
    let token = window.localStorage.getItem("jwt");
    console.log("the token is: ", token);

    // 将JWT添加到请求的Authorization头部
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  function (error) {
    // 请求错误处理
    return Promise.reject(error);
  }
);

const fetchedDataFiles = ref<any[]>([]);
// 获取用户历史文件
// const fetch_data = () => {
//   models_drawer.value = false
//   data_drawer.value = true

//   api.get('/fetch_datafiles/')
//     .then(response => {
//       let dataInfo = response.data
//       fetchedDataFiles.value.length = 0
//       for (let item of dataInfo) {
//         fetchedDataFiles.value.push(item)
//       }
//     })
//     .catch(error => {
//       console.log('fetch_datafiles_error: ',error)
//     })
// }

// 用户选择历史数据
const use_dataset = () => {};

const delete_dataset_confirm_visible = ref(false);
let row_dataset: any = null;
let index_dataset: any = null;
// 用户删除历史数据
const delete_dataset = (index_in: any, row_in: any) => {
  index_dataset = index_in;
  row_dataset = row_in;
  delete_dataset_confirm_visible.value = true;
};

const delete_dataset_confirm = () => {
  api
    .get("/delete_datafile?filename=" + row_dataset.dataset_name)
    .then((response: any) => {
      if (response.data.code == 200) {
        // 删除前端表中数据
        fetchedDataFiles.value.splice(index_dataset, 1);
        delete_dataset_confirm_visible.value = false;
        ElMessage({
          message: "文件删除成功",
          type: "success",
        });
      } else if (response.data.code == 400) {
        ElMessage({
          message: "删除失败: " + response.data.message,
          type: "error",
        });
      }
    })
    .catch((error) => {
      console.log("delete_datafile_error: ", error);
      ElMessage({
        message: "删除失败",
        type: "error",
      });
    });
};

const handleSwitchDrawer = (fetchData: any[]) => {
  models_drawer.value = false;

  fetchedDataFiles.value.length = 0;
  // fetchData.forEach(element => {
  //   fetchedDataFiles.value.push(element)
  // });
  for (let item of fetchData) {
    fetchedDataFiles.value.push(item);
  }
  nextTick(() => {
    console.log("nextTick");
    console.log("fetchedDataFiles: ", fetchedDataFiles.value);
  });

  data_drawer.value = true;

  // Object.assign(fetchedDataFiles, fetchData)
  console.log("fetchData: ", fetchData);
  console.log("fetchedDataFiles: ", fetchedDataFiles.value);
};
</script>

<style>
body {
  margin: 0;
}

.item {
  width: 150px;
  height: 50px;
  position: relative;
}

.deleteButton {
  position: absolute;
  top: 0px;
  right: 0px;
}

#source {
  border: 2px solid red;
}

#target {
  border: 2px solid blue;
}

.main {
  display: flex;
}

ul {
  list-style: none;
  padding-left: 0;
  width: 120px;
  background: #eee;
  text-align: center;
}

ul > li {
  height: 40px;
  line-height: 40px;
}

.main-right {
  border: 1px solid #ccc;
  flex: 1;
  margin-left: 15px;
  position: relative;
  background: #f4f4f4;
}

.node-info {
  position: relative;
  top: 5px;
}

.node-info-label {
  /* font-style: italic; */
  /* 垂直和水平居中的样式 */
  display: flex;
  align-items: center;
  justify-content: center;
  overflow-wrap: break-word;
  /* font-style: italic; */
  padding: 4px;
  position: relative;
  width: 112px;
  height: 80px;
  /* line-height: 36px; */
  font-size: 16px;
  text-align: center;
  border: 1px solid #e5e7eb;
  background: #fff;
  border-radius: 4px;
}

.node-info-label:hover {
  cursor: pointer;
  background: #f4eded;
}

.node-info-label:hover + .node-drag {
  /* background: red; */
  display: inline-block;
}

.node-drag {
  display: none;
  width: 10px;
  height: 10px;
  border-radius: 10px;
  background-color: red;
  border: 1px solid #ccc;
  position: absolute;
  right: -10px;
  top: 40px;
}

.node-drag:hover {
  display: inline-block;
}

.fullscreen_container {
  height: 100vh;
  /* display: flex; */
  /* flex-direction: column; */
}

/* .demo-tabs > .el-tabs__content {
  color: #6b778c;
  font-size: 32px;
  font-weight: 600;
} */

.result_visualization_view {
  width: 1200px;
  height: 600px;
  position: absolute;
}

.demo-tabs .custom-tabs-label span {
  vertical-align: middle;
  font-size: 16px;
  margin-left: 9px;
}

.status-indicator {
  position: fixed;
  top: 70px;
  right: 10px;
  padding: 5px 10px;
  border-radius: 5px;
  background-color: #f0ad4e;
  /* 初始颜色，如黄色 */
  color: white;
  z-index: 1000;
  /* 确保它显示在其他元素之上 */
}

/* 可以为不同的状态添加额外的类 */
.status-indicator.error {
  background-color: #d9534f;
  /* 红色表示错误或未通过检查 */
}

.status-indicator.success {
  background-color: #5cb85c;
  /* 绿色表示成功 */
}

.status-indicator.saved {
  background-color: #337ab7;
  /* 蓝色表示已保存 */
}

.status-indicator.rectify {
  background-color: #48a4a3;
  /* 表示正在修改模型 */
}

html,
body {
  height: 100vh;
  margin: 0;
  /* 移除默认的边距 */
  padding: 0;
  /* 移除默认的内边距 */
}

/* .el-main {
  background-color: #cee1f6;
  text-align: center;
  position: relative;
  
  background-image: url('./assets/可视化建模.png');
  background-position: center;
  background-size: contain;
  background-repeat: no-repeat;
} */
.el-main {
  background-color: #cee1f6;
  /*color: #333;*/
  text-align: center;
  position: relative;
  /* background-image: url('./assets/可视化建模.png'); */
  background-position: center;
  background-size: contain;
  /* height: 50vh; */
  background-repeat: no-repeat;
}

.has-background {
  background-color: #cee1f6;
  /*color: #333;*/
  text-align: center;
  position: relative;
  background-image: url("../assets/可视化建模.png");
  background-position: center;
  background-size: contain;
  /* height: 50vh; */
  background-repeat: no-repeat;
}

.clickable:hover {
  cursor: pointer;
  color: #007bff;
}
</style>
