<template>
  <!--
  更新页面 --v5.6
  版本  2024-7-19
  -->
  <div style="height: 100vh; overflow:hidden" @mouseover="background_IMG">
    <el-container class="fullscreen_container">
      <el-header style="height: 60px;text-align: center; line-height: 60px; position: relative;">
        <img src="./assets/logo.png" alt="" style="width: 50px; position: absolute; left: 5px; top: 5px;">
        <h2 style="font-size: 26px;">车轮状态分析与健康评估</h2>
        <div class="user-info-container" id="userInfo" style="position: absolute; right: 10px; top: 5px; color: white;">
          <span style="margin-right: 10px;">欢迎！ {{ username }}</span>
          <span @click="logout" class="clickable">退出登录</span>
        </div>
      </el-header>
      <el-container>
        <el-aside width="250px">
          <div style="font-size: 20px; font-weight: 700; background-color: #80a5ba; width: 250px; color: #f9fbfa;">
            算法选择区</div>
          <div style="background-color: #eff3f6; width: 250px;height: 600px;">
            <el-scrollbar height="600px" min-size="35" style="margin-left: 10px;">
              <el-column v-for="item in menuList2">
                <el-row><el-button style="width: 150px; margin-top: 10px; background-color: #4599be; color: white; "
                    icon="ArrowDown" @click="menu_details_second[item.label] = !menu_details_second[item.label]">
                    <el-text style="width: 105px; font-size: 15px; color: white;" truncated>{{ item.label
                      }}</el-text></el-button></el-row>

                <el-column v-if="menu_details_second[item.label]" v-for="option in item.options">

                  <el-row style="margin-left: 20px;">
                    <el-button style="width: 150px; margin-top: 7px; background-color: #75acc3;" icon="ArrowDown"
                      type="info" @click="menu_details_third[option.label] = !menu_details_third[option.label]">
                      <el-text style="width: 105px; font-size: 12px; color: white;" truncated>{{ option.label
                        }}</el-text></el-button>
                  </el-row>
                  <el-column v-if="menu_details_third[option.label]"
                    v-for="algorithm in Object.keys(option.parameters)">
                    <el-tooltip placement="right-start" :content="labels_for_algorithms[algorithm]" effect="light">
                      <div :draggable="true" @dragend="handleDragend($event, algorithm, option)" class="item"
                        @click="showIntroduction(algorithm.replace(/_multiple/g, ''))"
                        style="background-color:#f9fcff ; margin-top: 7px; width: 145px; height: 30px; margin-bottom: 10px; padding: 0px; border-radius: 5px; align-content: center; margin-left: 40px;">
                        <el-text style="width: 105px; font-size: 12px;" truncated>{{ labels_for_algorithms[algorithm]
                          }}</el-text>

                      </div>
                    </el-tooltip>


                  </el-column>
                </el-column>

              </el-column>
            </el-scrollbar>

          </div>
          <div style="font-size: 20px; font-weight: 700; background-color: #80a5ba; width: 250px; color: #f9fbfa;">
            加载数据
          </div>
          
          <uploadDatafile @switchDrawer="handleSwitchDrawer" :api="api"/>
          <div>已加载数据：{{ using_datafile }}</div>
        </el-aside>

        <el-main @dragover.prevent ref="efContainerRef" id="efContainer "
          style="height: auto; width: 600px; padding: 0px;">
          <div
            style=" position: relative; height: 32%; font-size: 20px; color: #003e50; font-weight: 500; font-family:Arial, Helvetica, sans-serif;  background-position: center;">
            <div id="statusIndicator" class="status-indicator">未建立模型</div>

            <!-- <el-button type="primary" style="font-size: 18px; width: 180px;" @click="drawer = true">打开功能区</el-button> -->

            <DraggableContainer :reference-line-visible="false">
              <Vue3DraggableResizable :draggable="true" :resizable="false" v-for="(item, index) in nodeList" 
                :key="item.nodeId" class="node-info" :id="item.id" :style="item.nodeContainerStyle"
                :ref="el => nodeRef[index] = el" @click="resultShow(item)">
                <el-popover placement="bottom" title="参数配置" :width="400" trigger="contextmenu">
                  <!-- 填参数 -->
                  <el-row v-if="item.use_algorithm != null && item.id != '1.2'"
                    v-for="(value, key) in item.parameters[item.use_algorithm]"
                    :key="item.parameters[item.use_algorithm].keys">
                    <el-col :span="8" style="align-content: center;"><span style="margin-left: 10px; font-size: 15px;">{{ labels_for_params[key] }}：</span></el-col>
                    <el-col :span="16"><el-input style="width:190px" :disabled="false" type="number"
                        v-model="item.parameters[item.use_algorithm][key]" /></el-col>
                    <!-- <el-input style="width:190px" :disabled="false"
                    v-model="item.parameters[item.use_algorithm][key]">
                        <template #prefix>
                          <div>
                            {{ labels_for_params[key] }}
                          </div>
                        </template>
                    </el-input> -->
                  </el-row>
                  <!-- 特征提取选择要显示的特征 -->
                  <el-row v-if="item.id == '1.2'">
                    <el-col :span="8" style="align-content: center;"><el-text style="margin-left: 10px; font-size: 15px;">选择特征：</el-text></el-col>
                    <el-col :span="16">
                      <div class="m-4">
                        <el-select v-model="features" multiple collapse-tags collapse-tags-tooltip
                          placeholder="选择需要提取的特征" :teleported="false">
                          <el-option v-for="(value, key) in item.parameters[item.use_algorithm]" :label="key"
                            :value="key" style="width: 200px; background-color: white;" />
                        </el-select>
                      </div>
                    </el-col>
                  </el-row>

                  <template #reference>
                    <div class="node-info-label el-dropdown-link font-style: italic;" :id=item.id>{{ item.label_display
                      }}
                      <div style="
                      position: absolute; left: 55px; top: 35px; width: 6px; height: 6px;
                      border: 2px solid #80a5ba; /* 边框颜色*/
                      border-radius: 50%; /* 设置为50%以创建圆形 */
                      background-color: transparent; /* 背景设置为透明，实现空心效果 */
                      /* 其他样式，如 cursor 可以设置拖拽时的鼠标光标形状 */
                      cursor: move; /* 鼠标悬停时显示可移动的光标 */" @mouseup="handleMouseup($event, item)"></div>
                      <el-button type="danger" icon="Delete" circle size="small" class="deleteButton"
                        @click="deleteNode(item.nodeId)" :disabled="modelSetup" />
                    </div>

                  </template>
                </el-popover>
                <!-- @contextmenu="params_setting(item[parameters])" -->
                <div class="node-drag" :id="item.id"></div>
              </Vue3DraggableResizable>
            </DraggableContainer>
            <!-- <el-upload action="" :auto-upload="false" v-model:file-list="file_list" :before-upload="checkFileType"
              limit="1" style="position: absolute; left: 320px; bottom: 20px;" :show-file-list="false">
              <el-button type="info" round style=" width: 125px; font-size: 17px; " icon="FolderAdd">
                上传数据
              </el-button>
            </el-upload> -->

            <!-- <template v-if="file_list.length > 0">
              <div class="custom-file-list" >
                <p v-for="file in file_list" :key="file.uid" style="display: flex; justify-content: space-between; align-items: center; ">
                  {{ file.name }}
                  
                  <el-button
                      style="float: right; width: 5px"
                      icon="close"
                      @click="handleDelete(file)">
                  </el-button>
                </p >
              </div>
            </template> -->
            <!-- <template v-if="file_list.length > 0">
              <div class="custom-file-list"
                style="display: flex; justify-content: space-between; align-items: center; ">
                <p v-for="file in file_list" :key="file.uid"
                  style="width: 180px; text-overflow: ellipsis; /* 当文本超出时显示省略号 */overflow: hidden; /* 隐藏超出部分的文本 */">
                  {{ file.name }}
                </p> -->
            <!-- 删除按钮 -->
            <!-- <el-button v-for="file in file_list" style="position: relative; float: right; width: 5px;" icon="close"
                  @click="handleDelete(file)">
                </el-button>
              </div>
            </template> -->
            <div
              style="position: absolute; right: 17px; bottom: 10px; width: 1430px; height: auto;display: flex; justify-content: space-between; align-items: center;">
              <el-button type="info" round style="width: 125px; font-size: 17px; background-color: #606266; "
                @click="fetch_models" icon="More">
                历史模型
              </el-button>
              <el-space size="large">
                
                <!-- <el-upload action="" :auto-upload="false" v-model:file-list="file_list" :before-upload="checkFileType"
                          limit="1" >
                    <el-button type="info" round style="position: absolute; width: 125px; font-size: 17px; "  icon="FolderAdd">
                      上传数据
                    </el-button>
                </el-upload> -->

                <el-button type="info" round style="width: 125px; font-size: 17px; background-color: #E6A23C;"
                  @click="handleclear" icon="Refresh">
                  清空模型
                </el-button>

                <!-- <el-button v-if="!to_rectify_model" type="info" :disabled="canCompleteModeling"
                  @mouseover="CompleteModeling" round style="width: 125px; font-size: 17px; background-color:  #722ED1;"
                  @click="finished_model" icon="Check">
                  完成建模
                </el-button> -->
                <el-button v-if="!to_rectify_model" type="primary" :disabled="canCompleteModeling"
                  @mouseover="CompleteModeling" round style="width: 125px; font-size: 17px;" @click="finished_model"
                  icon="Check">
                  完成建模
                </el-button>
                <!-- <el-button v-if="to_rectify_model" type="info" :disabled="canCompleteModeling"
                  @mouseover="CompleteModeling" round style="width: 125px; font-size: 17px; background-color:  #722ED1;"
                  @click="rectify_model" icon="Edit">
                  修改模型
                </el-button> -->
                <el-button v-if="to_rectify_model" type="primary" :disabled="canCompleteModeling"
                  @mouseover="CompleteModeling" round style="width: 125px; font-size: 17px;" @click="rectify_model"
                  icon="Edit">
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
                <el-button type="primary" :disabled="canCheckModel" @mouseover="checkModeling" round
                  style="width: 125px; font-size: 17px; " @click="check_model" icon="Search">
                  检查模型
                </el-button>

                <!-- <el-button type="info" :disabled="canSaveModel" @mouseover="saveModeling" round
                  style="width: 125px; font-size: 17px; background-color: #20A0FF;" @click="saveModelSetting(true, [])"
                  icon="Finished">
                  保存模型
                </el-button> -->
                <el-button type="primary" :disabled="canSaveModel" @mouseover="saveModeling" round
                  style="width: 125px; font-size: 17px;" @click="saveModelSetting(true, [])" icon="Finished">
                  保存模型
                </el-button>
                <!-- <el-button type="success" round style="width: 125px; font-size: 17px; background-color: #67C23A;"
                  @click="handleupload" icon="SwitchButton" :disabled="canStartProcess" @mouseover="startModeling">
                  开始运行
                </el-button> -->
                <el-button type="success" round style="width: 125px; font-size: 17px; " @click="run"
                  icon="SwitchButton" :disabled="canStartProcess || process_is_shutdown" @mouseover="startModeling">
                  开始运行
                </el-button>
                <el-button :disabled="canShutdown" type="danger" round style="width: 125px; font-size: 17px;"
                  @click="shutdown" icon="Close">
                  终止运行
                </el-button>
              </el-space>

            </div>
          </div>

          <div class="test" style="background-color: white;">
            <!--            <div-->
            <!--              style="position: relative; text-align: center; color: #333333; font-size: 20px; background-color: #333333; width: 100%; height: 22px;font-family:Arial, Helvetica, sans-serif; font-weight: 500;"-->
            <!--              v-loading="loading">-->
            <el-dialog v-model="dialogVisible" title="模型算法及参数设置" style="width: 1000px; height: 750px;">

              <el-tabs v-model="activeName" class="demo-tabs" @tab-click="handleClick">
                <el-tab-pane v-for="item in nodeList" :label="item.label" :name="item.nodeId">
                  <div v-if="item.label == '层次分析模糊综合评估'" style="position: relative; height: 630px; width: auto; ">
                    <div
                      style="position: absolute; left: 10px; top: 10px; height: 480px; width: 200px; background-color: aliceblue;">
                      <el-text size="large">指标层次构建</el-text>
                      <el-tree style="max-width: 600px" :data="dataSource" show-checkbox node-key="id"
                        default-expand-all :expand-on-click-node="false">
                        <template #default="{ node, data }">
                          <span class="custom-tree-node">
                            <span>{{ node.label }}</span>
                            <span>
                              <a @click="append(data)"> Append </a>
                              <a style="margin-left: 8px" @click="remove(node, data)"> Delete </a>
                            </span>
                          </span>
                        </template>
                      </el-tree>
                    </div>
                    <div
                      style="position: absolute; left: 230px; top: 10px; height: 480px; width: 720px; background-color: aliceblue;">
                      <el-text size="large">指标权重配置</el-text>

                      <div
                        style="position: absolute; top: 25px; left: 10px;width: 500px; height: 400px; background-color: white;">
                      </div>
                      <div
                        style="position: absolute; top: 25px; left: 500px; width: 190px; height: 400px; background-color: lightgray; margin-left: 20px;">
                        <el-space direction="vertical">
                          <el-text style="font-weight: bold; font-size: larger;">分值对照表</el-text>
                          <el-row><el-text style="font-size: medium;">1(1)同样重要</el-text></el-row>
                          <el-row><el-text style="font-size: medium;">2(1/2)稍稍微(不)重要</el-text></el-row>
                          <el-row><el-text style="font-size: medium;">3(1/3)稍微(不)重要</el-text></el-row>
                          <el-row><el-text style="font-size: medium;">4(1/4)稍比较(不)重要</el-text></el-row>
                          <el-row><el-text style="font-size: medium;">5(1/5)比较(不)重要</el-text></el-row>
                          <el-row><el-text style="font-size: medium;">6(1/6)稍非常(不)重要</el-text></el-row>
                          <el-row><el-text style="font-size: medium;">7(1/7)非常(不)重要</el-text></el-row>
                          <el-row><el-text style="font-size: medium;">8(1/8)稍绝对(不)重要</el-text></el-row>
                          <el-row><el-text style="font-size: medium;">9(1/9)绝对(不)重要</el-text></el-row>

                        </el-space>


                      </div>
                    </div>
                    <div
                      style="position: absolute; left: 10px; top: 500px; width: 940px; height: 125px; background-color: aliceblue;">
                    </div>
                  </div>
                  <!-- 选择具体算法以及设置参数 -->
                  <div v-if="item.label != '层次分析模糊综合评估'" style="margin-top: 5px; margin-left: 30px; width: 550px;">

                    <el-row>
                      <el-col :span="8"><el-text style="margin-top: 20px;">选择算法</el-text></el-col>
                      <el-col :span="16">
                        <el-select v-model="item.use_algorithm" placeholder="算法选择" popper-append-to-body="false"
                          id="select_algorithm" @change="setParams">
                          <el-option v-for="(value, key) in item.parameters" :label="labels_for_algorithms[key]"
                            :value="key" style="width: 225px; background-color: white; height: auto;">

                          </el-option>
                        </el-select>
                      </el-col>

                    </el-row>
                    <!-- 特征提取选择要显示的特征 -->
                    <el-row v-if="item.id == '1.3'">
                      <el-col :span="8">选择特征</el-col>
                      <el-col :span="16">
                        <div class="m-4">

                          <el-select v-model="features" multiple collapse-tags collapse-tags-tooltip
                            placeholder="选择需要提取的特征">
                            <el-option v-for="(value, key) in item.parameters[item.use_algorithm]" :label="key"
                              :value="key" style="width: 200px; background-color: white;" />
                          </el-select>
                        </div>
                      </el-col>

                    </el-row>
                    <el-row>
                      <el-text v-if="item.use_algorithm" style="margin-top: 1px; margin-bottom: 3px;">
                        算法介绍：{{ algorithm_introduction[item.use_algorithm] }}
                      </el-text>
                    </el-row>
                    <!-- 填参数 -->
                    <el-row v-if="item.use_algorithm != null && item.id != '1.3'"
                      v-for="(value, key) in item.parameters[item.use_algorithm]"
                      :key="item.parameters[item.use_algorithm].keys">
                      <!-- <el-col :span="8"><span>{{ labels_for_params[key] }}</span></el-col>
                      <el-col :span="16"><el-input style="width:190px" :disabled="false"
                          v-model="item.parameters[item.use_algorithm][key]" /></el-col> -->
                          <el-input style="width:190px" :disabled="false"
                            v-model="item.parameters[item.use_algorithm][key]">
                            <template #prefix>
                              <div>
                                {{ labels_for_params[key] }}
                              </div>
                            </template>
                          </el-input>
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
            <el-scollbar height="400px">
              <v-md-preview v-if="showPlainIntroduction" :text="introduction_to_show"
              style="text-align: left;"></v-md-preview>
              <v-md-preview v-if="showStatusMessage" :text="status_message_to_show"
                style="text-align: center;"></v-md-preview>
            </el-scollbar>
            
            <!-- <el-table id="progress" :v-loading="loading" element-loading-text="正在处理" :data="tableData" style="width: 100%; height: 250px ;" v-if="loading">
            </el-table> -->
            <!-- <span  :v-loading="loading" element-loading-text="正在处理" element-loading-background="rgba(122, 122, 122, 0.8)" style="width: 100%; height: 100%;"></span> -->
            <el-progress v-if="processing" :percentage="percentage" :indeterminate="true" />
            <iframe id='my_gradio_app' style="width: 1200px; height: 570px;" :src="refreshPardon" frameborder="0"
              v-if="isshow">
            </iframe>
            <!-- <div class="result_visualization_view" v-if="result_show">
              <el-scrollbar height="600px">
                <div v-if="display_health_evaluation">
                  <el-tabs class="demo-tabs" type="card" v-model="activeName1">
                    <el-tab-pane label="层级有效指标" name="first">
                      <img src=" " alt="figure1" id="health_evaluation_figure_1" />
                    </el-tab-pane>
                    <el-tab-pane label="指标权重" name="second">
                      <img src=" " alt="figure2" id="health_evaluation_figure_2" />
                    </el-tab-pane>
                    <el-tab-pane label="评估结果" name="third">
                      <el-col>
                        <img src=" " alt="figure3" id="health_evaluation_figure_3" />
                        <el-text :v-model="health_evaluation">{{ health_evaluation }}</el-text>
                      </el-col>

                    </el-tab-pane>
                  </el-tabs>
                </div>
              </el-scrollbar>
            </div> -->

            <el-scrollbar height="570px" v-if="result_show">
              <!-- 健康评估可视化 -->
              <el-tabs class="demo-tabs" type="card" v-model="activeName1" v-if="display_health_evaluation">
                <el-tab-pane label="层级有效指标" name="first">
                  <img :src="health_evaluation_figure_1" alt="figure1" id="health_evaluation_figure_1"
                    class="result_image" style="width: 900px; height: 450px;" />
                </el-tab-pane>
                <el-tab-pane label="指标权重" name="second">
                  <img :src="health_evaluation_figure_2" alt="figure2" id="health_evaluation_figure_2"
                    class="result_image" style="width: 900px; height: 450px;" />
                </el-tab-pane>
                <el-tab-pane label="评估结果" name="third">
                  <el-col>
                    <img :src="health_evaluation_figure_3" alt="figure3" id="health_evaluation_figure_3"
                      class="result_image" style="width: 900px; height: 450px;" />
                    <br>
                    <div style="width: 1000px; margin-left: 250px;">
                      <el-text :v-model="health_evaluation" style="font-weight: bold; font-size: 18px;">{{
                        health_evaluation
                      }}</el-text>
                    </div>
                  </el-col>
                </el-tab-pane>
              </el-tabs>
              <!-- 特征提取可视化 -->
              <el-table :data="transformedData" style="width: 100%; margin-top: 20px;"
                v-if="display_feature_extraction">
                <el-table-column v-for="column in columns" :key="column.prop" :prop="column.prop" :label="column.label"
                  :width="column.width">
                </el-table-column>
              </el-table>
              <!-- 特征选择可视化 -->
              <div v-if="display_feature_selection">
                <img :src="feature_selection_figure" alt="feature_selection_figure1" class="result_image"
                  style="width: 900px; height: 450px;" />
                <br>
                <div style="width: 1000px; margin-left: 250px;">
                  <el-text :v-model="features_selected" style="font-weight: bold; font-size: 20px;">选取特征结果为： {{
                    features_selected }}</el-text>
                </div>
              </div>
              <!-- 故障诊断可视化 -->
              <div v-if="display_fault_diagnosis" style="margin-top: 20px; font-size: 18px;">
                <div style="width: 1000px; margin-left: 250px; font-weight: bold">
                  故障诊断结果为： 由输入的振动信号，根据故障诊断算法得知该部件<span :v-model="fault_diagnosis"
                    style="font-weight: bold; color: red;">{{
                      fault_diagnosis }}</span>
                </div>

                <br>
                <img :src="fault_diagnosis_figure" alt="fault_diagnosis_figure" class="result_image"
                  style="width: 800px; height: 450px;" />
              </div>
              <!-- 故障趋势预测可视化 -->
              <div v-if="display_fault_regression" style="margin-top: 20px; font-size: 18px;">
                <div style="width: 1000px; margin-left: 250px;  font-weight: bold">
                  经故障诊断算法，目前该部件<span :v-model="fault_regression" style="font-weight: bold; color: red;">{{
                    fault_regression
                  }}</span>
                  <span v-if="!have_fault" :v-model="time_to_fault" style="font-weight: bold;">根据故障预测算法预测，该部件{{
                    time_to_fault
                  }}后会出现故障</span>
                </div>
                <br>
                <img :src="fault_regression_figure" alt="fault_regression_figure" class="result_image"
                  style="width: 800px; height: 450px;" />
              </div>
              <!-- 插值处理结果可视化 -->
              <div v-if="display_interpolation" style="margin-top: 20px; font-size: 18px;">
                <br>
                <img :src="interpolation_figure" alt="interpolation_figure" class="result_image"
                  style="width: 900px; height: 450px;" />
              </div>
              <!-- 无量纲化可视化 -->
              <div v-if="display_normalization" style="margin-top: 20px; font-size: 18px;">
                <div style="font-size: large;">原数据</div>
                <el-table :data="normalization_formdata_raw" style="width: 100%; margin-top: 20px;"
                >
                  <el-table-column v-for="column in normalization_columns" :key="column.prop" :prop="column.prop" :label="column.label"
                    :width="column.width">
                  </el-table-column>
                </el-table>
                <br>
                <div style="font-size: large;">标准化后数据</div>
                <el-table :data="normalization_formdata_scaled" style="width: 100%; margin-top: 20px;"
                >
                  <el-table-column v-for="column in normalization_columns" :key="column.prop" :prop="column.prop" :label="column.label"
                    :width="column.width">
                  </el-table-column>
                </el-table>
              </div>
              <!-- 小波降噪可视化 -->
              <div v-if="display_denoise">
                <img :src="denoise_figure" alt="denoise_figure" class="result_image"
                  style="width: 900px; height: 450px;" />
              </div>
            </el-scrollbar>

          </div>

        </el-main>


        <!-- 以抽屉的形式打开功能区 -->

        <el-drawer v-model="models_drawer" direction="ltr">

          <div style="display: flex; flex-direction: column; ">
            <el-col>

              <h2 style=" margin-bottom: 25px; color: #253b45;">历史模型</h2>

              <el-table :data="fetchedModelsInfo" height="500" stripe style="width: 100%">
                <el-popover placement="bottom-start" title="模型信息" :width="400" trigger="hover">
                </el-popover>
                <el-table-column :width="100" property="id" label="序号" />
                <el-table-column :width="150" property="model_name" label="模型名称" />
                <el-table-column :width="280" label="操作">
                  <template #default="scope">
                    <el-button size="small" type="primary" style="width: 50px;" @click="use_model(scope.row)">
                      使用
                    </el-button>
                    <el-button size="small" type="danger" style="width: 50px;"
                      @click="delete_model(scope.$index, scope.row)">
                      删除
                    </el-button>
                    <el-popover placement="bottom" :width='500' trigger="click">
                      <el-descriptions :title="model_name" :column="3" :size="size" direction="vertical"
                      >
                        <el-descriptions-item label="使用模块" :span="3">
                          <el-tag size="small" v-for="algorithm in model_algorithms">{{ algorithm }}</el-tag>
                        </el-descriptions-item>
                        <el-descriptions-item label="算法参数" :span="3">
                          <div v-for="item in model_params">{{ item.模块名 }}: {{ item.算法 }}</div>
                        </el-descriptions-item>
                      </el-descriptions>
                      <template #reference>
                        <el-button size="small" type="info" style="width: 80px" @click="show_model_info(scope.row)">
                          查看模型
                        </el-button>
                      </template>
                    </el-popover>
                  </template>
                </el-table-column>
              </el-table>

              <el-dialog v-model="delete_model_confirm_visible" title="提示" width="500">
                <span style="font-size: 20px;">确定删除该模型吗？</span>
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
                <el-table-column :width="150" property="dataset_name" label="文件名称" />
                <el-table-column :width="200" property="description" label="文件描述" />
                <el-table-column :width="200" label="操作">
                  <template #default="scope">
                    <el-button
                      size="small"
                      type="primary"
                      style="width: 50px"
                      @click="use_dataset(scope.row)"
                      :loading="loading_data"
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
      <el-footer style="height: 35px; position: relative;"><span
          style="position: absolute; top: -15px;">welcome!</span></el-footer>

    </el-container>
    <el-dialog v-model="dialogmodle" title="保存模型" draggable width="30%">
      <el-form :model="model_info_form">
        <el-form-item label="模型名称" :label-width='140'>
          <el-input style="width: 160px;" v-model="model_info_form.name" autocomplete="off" />
        </el-form-item>
      </el-form>
      <span class="dialog-footer">
        <el-button style="margin-left: 85px; width: 150px;" @click="dialogmodle = false">取消</el-button>
        <el-button style="width: 150px;" type="primary" @click="save_model_confirm">确定</el-button>
      </span>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>

import { onMounted, nextTick, ref } from 'vue'
import { jsPlumb } from 'jsplumb'
import { ElNotification, ElMessage } from "element-plus";
import axios from 'axios';
import { DraggableContainer } from "@v3e/vue3-draggable-resizable";
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import uploadDatafile from './uploadDatafile.vue';



const router = useRouter()

const dialogVisible = ref(false)

const activeName = ref('first')    // 控制标签页

const models_drawer = ref(false)
const data_drawer = ref(false)
const file_list = ref([])
// const Name = ref('xiaomao')

//控制按钮失效变量
const canStartProcess = ref(true)
// const canCompleteModeling = ref(true)
const canCompleteModeling = computed(() => {
  if (nodeList.value.length > 0 && !model_has_been_saved) {
    return false
  } else {
    return true
  }
})
const canCheckModel = ref(true)
const canSaveModel = ref(true)
const canShutdown = ref(true)

// { label: '添加噪声', id: '1.1', use_algorithm: null, parameters: { 'WhiteGaussianNoise': { SNR: -5 }, 'Noise': { param1: 10, param2: 20 } }, tip_show: false, tip: '为输入信号添加噪声' },

const menuList2 = ref([{
  label: '预处理算法', id: '1', options: [
    {
      label: '插值处理', id: '1.1', use_algorithm: null, parameters: {
        'neighboring_values_interpolation': {},
        'polynomial_interpolation': {},
        'bicubic_interpolation': {},
        'lagrange_interpolation': {},
        'newton_interpolation': {},
        'linear_interpolation': {}
      }, tip_show: false, tip: '对输入信号进行插值'
    },
    {
      label: '特征提取', id: '1.2', use_algorithm: null, parameters: {
        'time_domain_features': { 均值: false, 方差: false, 标准差: false, 偏度: false, 峰度: false, 四阶累积量: false, 六阶累积量: false, 最大值: false, 最小值: false, 中位数: false, 峰峰值: false, 整流平均值: false, 均方根: false, 方根幅值: false, 波形因子: false, 峰值因子: false, 脉冲因子: false, 裕度因子: false },
        'frequency_domain_features': { 重心频率: false, 均方频率: false, 均方根频率: false, 频率方差: false, 频率标准差: false, 谱峭度的均值: false, 谱峭度的峰度: false, },
        'time_frequency_domain_features': {
          均值: false, 方差: false, 标准差: false, 峰度: false, 偏度: false, 四阶累积量: false, 六阶累积量: false, 最大值: false, 最小值: false, 中位数: false, 峰峰值: false, 整流平均值: false, 均方根: false, 方根幅值: false, 波形因子: false, 峰值因子: false, 脉冲因子: false, 裕度因子: false,
          重心频率: false, 均方频率: false, 均方根频率: false, 频率方差: false, 频率标准差: false, 谱峭度的均值: false, 谱峭度的峰度: false,
        },
        'time_domain_features_multiple': { 均值: false, 方差: false, 标准差: false, 偏度: false, 峰度: false, 四阶累积量: false, 六阶累积量: false, 最大值: false, 最小值: false, 中位数: false, 峰峰值: false, 整流平均值: false, 均方根: false, 方根幅值: false, 波形因子: false, 峰值因子: false, 脉冲因子: false, 裕度因子: false },
        'frequency_domain_features_multiple': { 重心频率: false, 均方频率: false, 均方根频率: false, 频率方差: false, 频率标准差: false, 谱峭度的均值: false, 谱峭度的峰度: false, },
        'time_frequency_domain_features_multiple': {
          均值: false, 方差: false, 标准差: false, 峰度: false, 偏度: false, 四阶累积量: false, 六阶累积量: false, 最大值: false, 最小值: false, 中位数: false, 峰峰值: false, 整流平均值: false, 均方根: false, 方根幅值: false, 波形因子: false, 峰值因子: false, 脉冲因子: false, 裕度因子: false,
          重心频率: false, 均方频率: false, 均方根频率: false, 频率方差: false, 频率标准差: false, 谱峭度的均值: false, 谱峭度的峰度: false
        }
      }, tip_show: false, tip: '手工提取输入信号的特征'
    },
    {
      label: '无量纲化', id: '1.5', use_algorithm: null, parameters: {
        'max_min': {},
        'z-score': {},
        'robust_scaler': {},
        'max_abs_scaler': {}
      }, tip_show: false, tip: '对输入数据进行无量纲化处理'
    },
    {
      label: '特征选择', id: '1.3', use_algorithm: null, parameters: {
        'feature_imp': {num_features: 10},
        'mutual_information_importance': {num_features: 10},
        'correlation_coefficient_importance': {num_features: 10},
        'feature_imp_multiple': {num_features: 10},
        'mutual_information_importance_multiple': {num_features: 10},
        'correlation_coefficient_importance_multiple': {num_features: 10}
      }, tip_show: false, tip: '对提取到的特征进行特征选择'
    },
    {
      label: '小波变换', id: '1.4', use_algorithm: null, parameters: {
        'wavelet_trans_denoise': {}
      }, tip_show: false, tip: '对输入信号进行小波变换'
    }
  ], tip_show: false, tip: '包含添加噪声、插值以及特征提取等'
},
{
  label: '故障预测算法', id: '2', options: [
    {
      label: '故障诊断', id: '2.1', use_algorithm: null, parameters: {
        'random_forest': {},
        'svc': {},
        'gru': {},
        'lstm': {},
        'random_forest_multiple': {},
        'svc_multiple': {},
        'gru_multiple': {},
        'lstm_multiple': {}
      }, tip_show: false, tip: '根据提取特征对输入信号作故障诊断'
    },
    {
      label: '趋势预测', id: '2.2', use_algorithm: null, parameters: {
        'linear_regression': {},
        'linear_regression_multiple': {}
      }, tip_show: false, tip: '根据提取的信号特征对输入信号进行故障预测'
    }]
},
{
  label: '健康评估算法', id: '3', options: [
    {
      label: '层次分析模糊综合评估', id: '3.1', use_algorithm: null, parameters: {
        'FAHP': {},
        'FAHP_multiple': {}
      }, tip_show: false, tip: '将模糊综合评价法和层次分析法相结合的评价方法'
    }]
},
  // {
  //   label: '语音处理', id: '2', options: [{ label: '音频分离', id: '2.1', use_algorithm: null, parameters: { 'conformer': { num_workers: 8, layers: 64 }, 'sepformer': { num_workers: 16, layers: 64 } }, tip_show: false, tip: '可对输入的一维音频信号进行噪声分离' },
  //   { label: '声纹识别', id: '2.2', use_algorithm: null, parameters: { 'conformer': {}, 'lightweight_cnn_conformer': {} }, tip_show: false, tip: '根据输入的说话人语音识别说话人' }]
  // },

]);

const background_IMG = () => {
  if (nodeList.value.length == 0) {
    document.querySelector('.el-main').classList.add('has-background');

  }
  if (nodeList.value.length >= 1) {
    document.querySelector('.el-main').classList.remove('has-background');
    document.querySelector('.el-main').style.backgroundImage = ''

  }
}

// 算法推荐参数
const recommended_settings = {
  'time_domain_features': {
    '配置一': { 均值: true, 方差: true, 标准差: true, 偏度: true, 峰度: true, 四阶累积量: true, 六阶累积量: true, 最大值: true, 最小值: true, 中位数: true, 峰峰值: true, 整流平均值: true, 均方根: true, 方根幅值: true, 波形因子: true, 峰值因子: true, 脉冲因子: true, 裕度因子: true },
    'config_2': {}
  }
}

// 各个算法包含的参数对应的中文名
const labels_for_params = {
  SNR: '信噪比',
  layers: '网络层数',
  num_workers: '工作线程数',
  num_features: '选取特征数量',
}

const labels_for_algorithms = {
  'WhiteGaussianNoise': '高斯白噪声',
  'polynomial_interpolation': '多项式插值算法',
  'bicubic_interpolation': '双三次插值算法',
  'lagrange_interpolation': '拉格朗日插值算法',
  'newton_interpolation': '牛顿插值算法',
  'linear_interpolation': '线性插值算法',
  'time_domain_features': '单传感器时域特征提取',
  'frequency_domain_features': '单传感器频域特征提取',
  'time_frequency_domain_features': '单传感器时域和频域特征提取',
  'time_domain_features_multiple': '多传感器时域特征提取',
  'frequency_domain_features_multiple': '多传感器频域特征提取',
  'time_frequency_domain_features_multiple': '多传感器时域和频域特征提取',
  'FAHP': '单传感器层次分析模糊综合评估法',
  'FAHP_multiple': '多传感器分析模糊综合评估法',
  'feature_imp': '单传感器树模型的特征选择',
  'feature_imp_multiple': '多传感器树模型的特征选择',
  'mutual_information_importance': '单传感器互信息重要性特征选择',
  'mutual_information_importance_multiple': '多传感器互信息重要性特征选择',
  'correlation_coefficient_importance': '单传感器相关系数重要性特征选择',
  'correlation_coefficient_importance_multiple': '多传感器相关系数重要性特征选择',
  'random_forest': '单传感器随机森林故障诊断',
  'svc': '单传感器SVM的故障诊断',
  'gru': '单传感器GRU的故障诊断',
  'lstm': '单传感器LSTM的故障诊断',
  'random_forest_multiple': '多传感器随机森林故障诊断',
  'svc_multiple': '多传感器SVM的故障诊断',
  'gru_multiple': '多传感器GRU的故障诊断',
  'lstm_multiple': '多传感器LSTM的故障诊断',
  'wavelet_trans_denoise': '小波变换降噪',
  'max_min': 'max_min归一化',
  'z-score': 'z-score标准化',
  'robust_scaler': '鲁棒标准化',
  'max_abs_scaler': '最大绝对值标准化',
  'neighboring_values_interpolation': '邻近值插补算法',
  'linear_regression': '单传感器线性回归趋势预测',
  'linear_regression_multiple': '多传感器线性回归趋势预测'
}


const algorithm_introduction = {
  'WhiteGaussianNoise': '高斯白噪声(White Gaussian Noise)在通信、信号处理及科学研究等多个领域中扮演着重要角色。它作为一种理想的噪声模型，具有独特的统计特性和功率谱分布，为系统性能评估、算法测试及信号分析提供了有力工具',
  'polynomial_interpolation': '多项式补插法是一种分段三次Hermite插值方法,它在每个数据段上使用三次多项式来逼近数据点,并且在连接处保持一阶导数的连续性。与双三次插值不同,' +
    '多项式插值在每个段上使用不同的三次多项式,并且尝试保持二阶导数的变号,从而生成一个形状类似于原始数据的曲线。',
  'bicubic_interpolation': '双三次插值是一种平滑插值方法,它通过三次多项式段来逼近数据点,并且在每个数据段的连接处保持一阶导数和二阶导数的连续性。' +
    '这种方法可以生成一个平滑的曲线,通过数据点,并且在数据点处具有连续的一阶和二阶导数。',
  'lagrange_interpolation': '对于所给数据中每一行每一列空白的位置,取空白位置上下3个相邻的值作为输入依据,' +
    '根据拉格朗日算法构建一个多项式函数,使得该多项式函数在取得的这些点上的值都为零,' +
    '将空白位置的行值作为输入,计算出y值,替换原来的空白值,从而达到插值的效果。',
  'newton_interpolation': '在牛顿插值法中,首先利用一组已知的数据点计算差商,再将差商带入插值公式f(x)。将所提供数据中的数据各属性值作为y,而将索引号定义为x,' +
    '对于所给数据中每一行每一列空白的位置,取空白位置上下4个相邻的值作为输入依据,' +
    '并计算差商再反向带入包含差商的插值公式,替换原来的空白值。',
  'linear_interpolation': '在线性插值算法中,首先遍历数据中的每一行每一列，找到空值位置并获取相邻点的值,' +
    '然后去除相邻的空值,并处理边界情况,计算插值结果,替换原来的空白值。',
  'time_domain_features': '时域特征提取是一种从信号中直接提取其在时间轴上特性的技术。它主要用于从原始信号中提取出诸如幅度、频率、周期、波形等关键信息，以便于后续的信号处理和分析。',
  'frequency_domain_features': '频域特征提取是将时域信号转换到频域后，提取其频域上的特征。它主要用于从复杂的时域信号中提取出与频率相关的有用信息。',
  'time_frequency_domain_features': '提取信号的时域和频域特征，可以结合时域与频域特征各自的优点',
  'FAHP': '层次分析模糊综合评估法是一种将模糊综合评价法和层次分析法相结合的评价方法，在体系评价、效能评估，系统优化等方面有着广泛的应用，是一种定性与定量相结合的评价模型，一般是先用层析分析法确定因素集，\
           然后用模糊综合评判确定评判效果。模糊法是在层次法之上，两者相互融合，对评价有着很好的可靠性'
}

// const plain_introduction = {
//   '添加噪声': '### 添加噪声算法可对输入信号添加如高斯噪声等常见的噪声\n***\n ### 添加噪声的主要应用有：\n **1. 增强信号检测。在某些特定的非线性系统中，噪声的存在能够增强微弱信号的检测能力，这种现象被称为随机共振。**\n \
//   **2. 减小重构误差。如果在信号处理中只加入正的白噪声，那么在重构过程中可能会多出来加入的白噪声，从而增大了重构误差。因此，加入正负序列的白噪声可以在分解过程中相互抵消，减小重构误差。**\n### 本模块包含算法如下:\n **高斯白噪声： 高斯白噪声(White Gaussian Noise)在通信、信号处理及科学研究等\
//   多个领域中扮演着重要角色。它作为一种理想的噪声模型，具有独特的统计特性和功率谱分布，为系统性能评估、算法测试及信号分析提供了有力工具**',
//   '插值处理': '### 插值处理算法可以对输入信号进行插值操作\n***\n ### 插值处理的主要应用有：\n **1. 数值计算。函数逼近：插值方法，如拉格朗日插值多项式，可以用于在给定节点上逼近任意函数，从而在节点外的位置计算函数的近似值。**\n \
//   **2. 图像处理。图像放大与缩小：双线性插值等方法可以用于图像的放大或缩小，以获取更高分辨率或适当尺寸的图像。图像平滑：线性插值等方法可以用于平滑图像中的噪声或异常值，提高图像质量。**\
//   **3. 数据净化。缺失值处理：插值法是一种有效的缺失值处理方法，能够根据不同情况选择合适的插值类型进行估算，实现数据的完整性和连续性。数据平滑：插值法可以用于构建连续且平滑的函数模型来拟合数据，消除噪声和异常值的影响，提高数据质量。**\n \
//   ### 本插值处理模块中包含算法：\n **多项式插值算法、双三次插值算法、拉格朗日插值算法、牛顿插值算法、线性插值算法**',
//   '特征提取': '### 特征提取算法可对输入信号进行人工特征提取\n***\n ### 特征提取算法中主要包括信号时域特征和频域特征的提取：\n **1. 时域特征。定义：时域特征描述的是信号随时间变化的关系。在时域中，信号被表示为时间的函数，其动态信号描述信号在不同时刻的取值。\
//   特点：时域表示较为形象与直观，对于正弦信号等简单波形的时域表达，可以通过幅值、频率、相位三个基本特征值唯一确定。时域分析能够直接反映信号随时间的实时变化。**\n **2. 频域特征。定义：频域特征描述的是信号在频率方面的特性。\
//   频域分析是通过对信号进行傅立叶变换等数学方法，将信号从时间域转换到频率域，从而研究信号的频率结构。特点：频域分析能够深入剖析信号的本质，揭示信号中不易被时域分析发现的特征。频域特征通常用于表达信号的周期性信息。**\n ### 本特征提取算法中提取的时域和频域的特征包括：\
//   \n **1. 时域特征：均值、方差、标准差、峰度、偏度、四阶累积量、六阶累积量、最大值、最小值、中位数、峰峰值、整流平均值、均方根、方根幅度、波形因子、峰值因子、脉冲因子、裕度因子**\n \
//   **1. 频域特征：重心频率、均方频率、均方根频率、频率方差、频率标准差、谱峭度的均值、谱峭度的标准差、谱峭度的峰度、谱峭度的偏度**',
//   '层次分析模糊综合评估': '### 层次分析模糊综合评估法是一种将模糊综合评价法和层次分析法相结合的评价方法\n***\n ### 算法优点：\n **1. 可以综合考虑多个因素的影响，给出全面评价结果。**\n \
//   **2. 评价结果是一个矢量，而不是一个点值，包含的信息比较丰富，既可以比较准确的刻画被评价对象，又可以进一步加工，得到参考信息。**\n **3. 模糊评价通过精确的数字手段处理模糊的评价对象，能对蕴藏信息呈现模糊性的资料作出比较科学、合理、贴近实际的量化评价。**'
// }

const plain_introduction = {
  'polynomial_interpolation': '# 多项式插值方法\n' +
    '## 多项式插值是一种数学技术，通过构造一个多项式来精确地通过一组给定的数据点，从而对数据进行平滑逼近。\n' +
    '# 使用场景\n' +
    '### 1. **平滑逼近**：适用于需要对数据点进行精确插值的情况，以发现数据之间的潜在趋势。\n' +
    '### 2. **曲线拟合**：在数据点之间建立一个连续的多项式曲线，用于表示数据的整体形态。\n' +
    '### 3. **信号重建**：在信号采集中，用于填补缺失的数据点，恢复信号的完整性。\n' +
    '### 4. **滤波和去噪**：在信号处理中，多项式插值可以帮助平滑信号，减少噪声的影响。',
  'neighboring_values_interpolation': '# 邻近值插补在数据预处理中的应用\n' +
    '## 邻近值插补是一种基于数据点之间相似性或距离的数据插补方法，用于处理缺失数据。它通过选择缺失值附近已有的值来填补空缺。\n' +
    '# 插补的使用场景\n' +
    '### 1. 时间序列数据：在时间序列分析中，当数据集中出现缺失值时，可以使用邻近时间点的值进行插补。\n' +
    '### 2. 快速预处理：在需要快速进行数据预处理，而没有时间或资源进行更复杂的插补方法时，邻近值插补是一种实用的选择。\n' +
    '### 3. 异常值处理：当缺失值可能是由于数据收集过程中的异常或错误造成时，邻近值插补可以作为一种简单的错误校正方法。\n' +
    '### 4. 数据预处理：在数据预处理阶段，邻近值插补可以用于填补由于数据录入错误或丢失造成的缺失值。\n' +
    '### 5. 缺失数据不多的情况：当数据集中的缺失值不多，且分布均匀时，邻近值插补提供了一种快速且有效的解决方案。',
  'bicubic_interpolation': '# 双三次插值方法\n' +
    '## 双三次插值是一种高效的插值技术，它通过构造一个在数据点及其一阶和二阶导数上都连续的三次多项式来逼近数据。\n' +
    '# 使用场景\n' +
    '### 1. **高精度逼近**：适用于需要在数据点之间进行平滑且高精度插值的情况。\n' +
    '### 2. **曲面建模**：在二维数据集中，用于创建连续的曲面模型，以便于分析和可视化。\n' +
    '### 3. **高质量信号重建**：在信号采集中，用于填补缺失的数据点，同时保持信号的平滑性和连续性。\n' +
    '### 4. **图像处理**：在图像缩放和旋转中，双三次插值可以减少锯齿效应，保持图像质量。\n',
  'lagrange_interpolation': '# 拉格朗日插值法\n' +
    '## 拉格朗日插值法是一种多项式插值方法，通过构造一个多项式来精确匹配一组给定的数据点。\n' +
    '# 使用场景\n' +
    '### 1. **精确逼近**：适用于需要在数据点之间进行精确插值的情况，尤其是在数据点数量较少时。\n' +
    '### 2. **曲线拟合**：在数据点之间建立一个多项式曲线，用于模拟数据的整体趋势。\n' +
    '### 3. **信号重建**：在信号采集中，用于填补丢失的数据点，恢复信号的完整性。\n' +
    '### 4. **数据平滑**：在信号处理中，拉格朗日插值可以帮助平滑信号，减少噪声的影响。\n',
  'newton_interpolation': '# 牛顿插值法\n' +
    '## 牛顿插值法是一种高效的多项式插值技术，它利用差商构建一个多项式，能够通过一组给定的数据点。\n' +
    '# 使用场景\n' +
    '### 1. **递归构建**：适用于需要逐步增加数据点进行插值的情况，便于更新和维护多项式模型。\n' +
    '### 2. **曲线拟合**：在数据点之间构建一个多项式曲线，用于模拟数据的趋势和模式。\n' +
    '### 3. **动态信号重建**：在信号处理中，牛顿插值可以动态地填补丢失的数据点，适应信号变化。\n' +
    '### 4. **实时数据处理**：适用于需要实时处理和更新数据的场合，如在线信号分析。\n',
  'linear_interpolation': '# 线性插值方法\n' +
    '## 线性插值是一种基本的插值方法，通过在两个已知数据点之间构建一条直线来估计未知数据点的值。\n' +
    '# 使用场景\n' +
    '### 1. **简单估算**：在需要快速且直接的数据点估计时使用，适用于数据变化趋势为线性的情况。\n' +
    '### 2. **趋势分析**：用于识别和展示数据的线性关系，便于理解数据的基本情况。\n' +
    '### 3. **数据填补**：在信号采集中，用于填补因测量误差或数据丢失造成的空白。\n' +
    '### 4. **去噪处理**：在信号处理中，线性插值可以用于简化信号，减少高频噪声。\n',
  'time_domain_features': '# 时域特征提取\n' +
    '## 时域特征反映了信号在时间维度上的特性，不涉及任何频率转换，如傅里叶变换。而时域特征提取是从信号的原始时间序列数据中抽取关键信息的过程，这些信息能够表征信号的基本属性和内在特性。\n' +
    '# 主要方法\n' +
    '### 1. **峰值检测**：识别信号的最大或最小值点。\n' +
    '### 2. **统计分析**：计算信号的均值、方差、偏度、峭度等统计量。\n' +
    '### 3. **能量计算**：评估信号的总能量，通常通过对信号平方后积分。\n' +
    '### 4. **时间参数测量**：测量信号的周期、持续时间、延迟等。\n' +
    '### 5. **波形特征分析**：提取波形的特定形状特征，如脉冲宽度、上升时间等。\n' +
    '### 6. **相关性分析**：计算信号与参考信号之间的相关度。\n' +
    '### 7. **自相关函数**：分析信号在不同时间延迟下的相关性。\n',
  'frequency_domain_features': '# 频域特征提取\n' +
    '## 频域特征提取是一种分析技术，它通过将信号从时域转换到频域来提取信号的频率成分，进而分析和处理信号。\n' +
    '# 主要方法\n' +
    '### 1. **傅里叶变换(FT)**：将时域信号转换为频域表示。\n' +
    '### 2. **短时傅里叶变换(STFT)**：分析时变信号的局部频率特性。\n' +
    '### 3. **小波变换(WT)**：提供时间和频率的局部化信息，适用于非平稳信号分析。\n' +
    '### 4. **谱估计技术**：如周期图法、协方差法等，用于更精细的频谱分析。\n',
  'time_frequency_domain_features': '# 时频域特征提取\n' +
    '## 时频域特征提取结合了时域和频域的方法来研究信号的局部特性，时频域分析不仅关注信号在单一时间点的特征，也关注信号在不同时间段的频率变化，适用于分析非平稳信号。\n' +
    '# 主要方法\n' +
    '### 1. **短时傅里叶变换(STFT)**：通过在不同时间窗口上应用傅里叶变换来分析信号的局部频率特性。\n' +
    '### 2. **小波变换(WT)**：利用小波函数来分析信号在不同时间和频率尺度上的特性。\n' +
    '### 3. **Wigner-Ville分布**：一种二维时频表示，能够展示信号的频率和时变特性。\n' +
    '### 4. **Hilbert-Huang变换(HHT)**：结合经验模态分解(EMD)和Hilbert变换，用于分析非线性和非平稳信号。\n',
  'FAHP': '# 层次分析模糊综合评估法\n' +
    '## 层次分析模糊综合评估法是一种将模糊综合评价法和层次分析法相结合的评价方法，在体系评价、效能评估，系统优化等方面有着广泛的应用，是一种定性与定量相结合的评价模型，一般是先用层析分析法确定因素集，然后用模糊综合评判确定评判效果。模糊法是在层次法之上，两者相互融合，对评价有着很好的可靠性。\n' +
    '# 评价过程\n' +
    '### 1. **建立层次结构模型**：首先使用层次分析法确定问题的目标层、准则层和方案层。\n' +
    '### 2. **成对比较和一致性检验**：通过成对比较确定各因素的相对重要性，并进行一致性检验。\n' +
    '### 3. **确定权重向量**：计算准则层和方案层的权重向量。\n' +
    '### 4. **构建模糊评价模型**：利用模糊综合评价法构建评价模型，确定评价指标的隶属度函数。\n' +
    '### 5. **模糊综合评判**：综合考虑各因素的权重和隶属度，进行模糊综合评判，得出最终的评价结果。\n' +
    '\n',
  'FAHP_multiple': '# 层次分析模糊综合评估法\n' +
    '## 层次分析模糊综合评估法是一种将模糊综合评价法和层次分析法相结合的评价方法，在体系评价、效能评估，系统优化等方面有着广泛的应用，是一种定性与定量相结合的评价模型，一般是先用层析分析法确定因素集，然后用模糊综合评判确定评判效果。模糊法是在层次法之上，两者相互融合，对评价有着很好的可靠性。\n' +
    '# 评价过程\n' +
    '### 1. **建立层次结构模型**：首先使用层次分析法确定问题的目标层、准则层和方案层。\n' +
    '### 2. **成对比较和一致性检验**：通过成对比较确定各因素的相对重要性，并进行一致性检验。\n' +
    '### 3. **确定权重向量**：计算准则层和方案层的权重向量。\n' +
    '### 4. **构建模糊评价模型**：利用模糊综合评价法构建评价模型，确定评价指标的隶属度函数。\n' +
    '### 5. **模糊综合评判**：综合考虑各因素的权重和隶属度，进行模糊综合评判，得出最终的评价结果。\n' +
    '\n',
  'feature_imp': '# 使用树模型进行特征选择\n' +
    '\n' +
    '## 特征选择是机器学习中的一项关键任务，用于识别最有信息量的特征，以提高模型的性能和可解释性。树模型，包括决策树、随机森林和梯度提升树等，提供了多种特征选择的方法。特征选择旨在从原始特征集中挑选出对模型预测最有用的特征子集。在树模型中，这一过程通常基于以下原理：\n' +
    '\n' +
    '### 1. **分裂准则**：树模型在分裂节点时选择特征，基于如信息增益、基尼不纯度等准则。\n' +
    '### 2. **特征重要性**：衡量特征在模型中的贡献度，对模型性能的影响.\n' +
    '### 3. **数据驱动**：特征选择过程完全基于数据和模型的反馈，而不是基于领域知识或其他外部标准。',
  'mutual_information_importance': '# 互信息重要性特征选择\n' +
    '## 互信息是度量两个随机变量之间相互依赖性的统计量，它在特征选择中用于识别最有信息量的特征。互信息重要性特征选择遵循以下原理：\n' +
    '### 1. 最大化信息量：选择那些包含关于目标变量最多信息的特征。\n' +
    '### 2. 减少冗余： 避免选择相互之间提供重复信息的特征。\n' +
    '### 3. 计算效率: 采用有效的算法来计算互信息，以保证特征选择过程的可行性。\n' +
    '### 4. 稳健性; 确保特征选择方法对于数据的小变化是稳健的。\n' +
    '### 5. 适应性： 特征选择方法应能适应不同类型的数据分布。\n' +
    '### 6. 降维： 通过特征选择减少特征空间的维度，以简化模型并提高计算效率。\n' +
    '### 7. 模型无关：互信息特征选择是模型无关的，即它不依赖于特定的预测模型。\n' +
    '### 8. 增量学习：在新数据到来时，能够更新特征选择结果，适应数据的变化。\n',
  'correlation_coefficient_importance': '# 相关系数重要性特征选择\n' +
    '## 相关系数特征选择是一种评估特征与目标变量之间线性关系强度的方法。其中相关系数是量化两个变量之间线性关系的统计度量，通常使用皮尔逊相关系数。其遵循以下原理：\n' +
    '### 1. 线性关系：假设特征与目标变量之间存在线性关系，通过相关系数评估这种关系的强度。\n' +
    '### 2. 特征有效性：选择那些与目标变量具有显著线性相关性的特征，以提高模型的预测能力。\n' +
    '### 3. 多重共线性控制： 在选择特征时，避免纳入高度线性相关的特征，减少模型的多重共线性问题。\n' +
    '### 4. 简洁性原则：倾向于选择较少的特征，以简化模型结构，提高模型的可解释性和泛化能力。\n' +
    '### 5. 计算效率： 相关系数的计算相对简单快速，适用于大规模数据集的特征选择。\n' +
    '### 6. 稳健性：考虑特征选择方法对于数据中的异常值和噪声的敏感性，确保选择结果的稳定性。\n' +
    '### 7. 模型无关性：虽然基于线性关系选择特征，但所选特征适用于不同类型的预测模型。\n' +
    '### 8. 适应性：特征选择方法应能够适应不同的数据特征和分布情况，保持选择结果的准确性。',
  'random_forest': '# 随机森林故障诊断\n' +
    '## 随机森林是一种集成学习方法，广泛应用于故障诊断，能够处理复杂的数据模式和识别多种故障类型。其中随机森林是由多个决策树构成的集成模型，每棵树在数据集的不同子集上训练，并对结果进行投票或平均以得出最终预测。\n' +
    '# 特点\n' +
    '### 1.高准确性：集成多个决策树的预测结果，提高整体的诊断准确性。\n' +
    '### 2.鲁棒性：对数据中的噪声和异常值具有较好的抵抗力。\n' +
    '### 3.多故障类型识别：能够同时处理和识别多种不同类型的故障。\n' +
    '### 4.捕捉非线性关系;有效识别数据中的非线性故障模式和复杂关系。\n' +
    '### 5.模型泛化能力:由于集成了多个树，随机森林具有良好的泛化能力，减少过拟合风险。\n',
  'svc': '# 支持向量机（SVM）故障诊断\n' +
    '## 支持向量机是一种在故障诊断中广泛使用的监督学习模型，以其在分类和模式识别任务中的强大性能而著称。支持向量机是一种基于间隔最大化原则来构建分类器的方法，特别适用于高维数据和非线性问题。\n' +
    '# 特点\n' +
    '### 1.高维数据处理能力:SVM通过核技巧有效地处理高维数据，无需显式地映射到高维空间。\n' +
    '### 2.间隔最大化:SVM通过最大化数据点之间的间隔来提高分类的鲁棒性。\n' +
    '### 3.核函数:使用不同的核函数（如线性核、多项式核、径向基函数核等）来处理线性不可分的数据。\n' +
    '### 4.软间隔引入:允许一些数据点违反间隔，以提高模型的泛化能力。\n' +
    '### 5.正则化:通过正则化项控制模型复杂度，防止过拟合。\n' +
    '### 6.多类分类:通过策略如一对多（OvR）方法，SVM能够处理多类故障诊断问题。\n',
  'gru': '# 门控循环单元（GRU）故障诊断\n' +
    '## 门控循环单元是一种特殊的循环神经网络，适用于序列预测和时间序列分析的递归神经网络结构，它引入了更新门和重置门机制，以改善梯度流动并捕捉长期依赖关系，特别适用于故障诊断任务。\n' +
    '# 特点\n' +
    '### 1. 长期依赖学习能力：GRU特别设计了更新门来解决传统RNN中的梯度消失问题，使其能够学习长期依赖信息。\n' +
    '### 2. 动态时间序列处理能力:GRU能够处理时间序列数据中的动态变化，适用于捕捉故障发生前后的模式变化。\n' +
    '### 3. 门控机制的灵活性:通过更新门和重置门的控制，GRU可以灵活地决定信息的保留和遗忘，以适应不同的故障特征。\n' +
    '### 4. 易于集成和训练:GRU模型易于在现有的深度学习框架中实现和训练，便于与其他模型或数据处理流程集成。\n',
  'lstm': '# 长短期记忆网络（LSTM）故障诊断\n' +
    '## 长短期记忆网络是一种特殊类型的循环神经网络（RNN），设计用来解决传统RNN在处理长序列数据时的梯度消失问题。LSTM因其出色的记忆和遗忘机制，在序列预测和时间序列分析中表现卓越，非常适合故障诊断任务。\n' +
    '# 特点\n' +
    '### 1. 有效的长期依赖处理:LSTM通过其复杂的门控机制（输入门、遗忘门、输出门）来控制信息的流动，有效捕捉和记忆长期依赖关系。\n' +
    '### 2. 强大的序列预测能力:LSTM能够分析时间序列数据中的复杂模式，预测故障发生的概率和时间点。\n' +
    '### 3. 适应性强的门控机制:通过遗忘门和输入门的协同工作，LSTM可以决定哪些信息应该被遗忘，哪些信息应该被更新和保留。\n' +
    '### 4. 良好的泛化能力:经过适当的训练，LSTM可以学习到数据中的深层特征，对未见过的故障模式具有良好的泛化和识别能力。\n',
  'wavelet_trans_denoise': '# 小波变换去噪\n' +
    '## 小波变换去噪是一种利用小波分析对信号进行降噪处理的方法，它通过将信号分解为不同时间尺度上的成分，然后有选择地去除噪声成分。\n' +
    '# 基本原理\n' +
    '### 1. 多尺度分解：小波变换将信号分解为不同时间尺度（或频率）上的成分，这些成分称为小波系数。\n' +
    '### 2. 信号与噪声的分：信号往往在小波变换的低频部分具有较大的系数，而噪声则在高频部分较为显著。\n' +
    '### 3. 阈值处理：对小波系数进行阈值处理，设置一个阈值，将小于该阈值的系数视为噪声并置零或进行缩减，而保留较大的系数。\n' +
    '### 4. 重构信号:通过保留和放大重要的小波系数，忽略或减弱噪声成分，然后通过小波逆变换重构出降噪后的信号。\n',
  'max_min': '# 最大最小值归一化（Max-Min Normalization）\n' +
    '## 最大最小值归一化通过一个简单的线性变换过程，将数据特征的值缩放到[0, 1]的范围内。这个过程包括识别数据集中每个特征的最大值和最小值，然后利用这两个值来调整所有数据点，确保最小的数据点映射到0，最大的数据点映射到1，而其他点则根据它们与最小值和最大值的关系被映射到(0, 1)区间内。如果需要不同的数值范围，可以通过额外的缩放和平移操作来实现。这种方法易于实现且计算效率高，但要注意它对数据中的极端值或异常值较为敏感。\n' +
    '# 特点\n' +
    '### 简单性：最大最小值归一化方法简单，易于实现。\n' +
    '### 快速性：计算过程快速，适合大规模数据集。\n' +
    '### 数据分布敏感：归一化结果依赖于数据中的最小值和最大值，对异常值敏感。',
  'z-score': '# z-score标准化\n' +
    '## z-score标准化是一种数据预处理技术，用于将数据转换为具有平均值为0和标准差为1的标准分数。这种转换基于原始数据的均值和标准差，使得转换后的数据分布更加规范化，便于比较和分析。\n' +
    '# 特点\n' +
    '### 1. 中心化和尺度统一：数据通过减去均值并除以标准差进行转换，结果是一个中心化在0，单位标准差的分布。\n' +
    '### 2. 正态分布适配：该方法假设数据近似正态分布，通过转换使得数据更接近标准正态分布。\n' +
    '### 3. 异常值敏感性较低：与基于极端值的方法不同，z-score标准化使用均值和标准差，因此对异常值的影响较小。\n' +
    '### 4. 易于解释性：转换后的z-score值表示数据点距离均值的标准差数，提供了数据点分布情况的直观度量。\n',
  'robust_scaler': '# 鲁棒标准化\n' +
    '## 鲁棒标准化是一种数据预处理技术，它使用数据的中位数和四分位数范围（IQR）来缩放数据，从而对异常值具有较高的抵抗力。这种方法不依赖于数据的均值和标准差，而是使用中位数和IQR来确定数据的尺度。\n' +
    '# 特点\n' +
    '### 1. 对异常值的鲁棒性：鲁棒标准化通过使用中位数和IQR，减少了异常值对数据缩放的影响。\n' +
    '### 2. 中位数和四分位数：数据的中心位置由中位数确定，而数据的尺度由IQR（即第三四分位数与第一四分位数之差）确定。\n' +
    '### 3. 缩放方法：鲁棒标准化通常涉及将数据点减去中位数，然后除以IQR的一定比例（通常是1/0.7413，这个值使得IQR近似等于标准差的1倍）。\n',
  'max_abs_scaler': '# 最大绝对值标准化\n' +
    '## 最大绝对值标准化是一种数据预处理技术，通过将数据的每个特征的值除以其最大绝对值来实现标准化。这种方法不关心数据的正负符号，只关注值的大小。\n' +
    '# 特点\n' +
    '### 1. 简单易行：最大绝对值标准化的计算过程简单，易于实现和理解。\n' +
    '### 2. 忽略数据符号：该方法只考虑数据的绝对值，因此对数据的正负符号不敏感。\n' +
    '### 3. 抵抗异常值：由于只依赖于最大绝对值，该方法对异常值具有一定的抵抗力。\n' +
    '### 4. 缩放到[-1, 1]区间：标准化后的数据将位于[-1, 1]区间内，便于比较不同特征的值。',
  'linear_regression': '# 线性回在趋势预测\n' +
    '## 线性回归是一种用于确定因变量（预测目标）与一个或多个自变量（预测因子）之间线性关系的方法。在趋势预测中，线性回归模型通过拟合历史数据来预测未来的趋势或模式。\n' +
    '# 特点\n' +
    '### 1. 模型表达性：线性回归模型通过直线（单变量线性回归）或平面/超平面（多元线性回归）来表达数据之间的关系。\n' +
    '### 2. 预测连续变量：主要用于预测连续变量，如房价、气温、销售额等。\n' +
    '### 3. 结合定性变量：通过哑变量（Dummy Variables）可以包含定性变量，以研究它们对趋势的影响。\n' +
    '### 4. 基于最小二乘法：通常使用最小二乘法来估计模型参数，这种方法可以找到最佳拟合直线或超平面。\n' +
    '### 5. 假设正态分布：在简单线性回归中，假设误差项呈正态分布，且具有常数方差。\n' +
    '### 6. 稳健性分析：虽然对异常值敏感，但通过残差分析和杠杆值可以识别并处理异常值和高影响力点。\n' +
    '### 7. 可扩展性：从简单的单变量线性回归可以扩展到包含多个预测变量的多元线性回归。\n',
  // '### 8. 包含交互作用和多项式项：可以通过添加交互作用项和多项式项来提高模型的预测能力。\n',
  '随机森林趋势预测': '# 随机森在趋势预测\n' +
    '## 随机森林是一种集成学习方法，它通过构建多个决策树并将它们的预测结果进行汇总，以提高模型的准确性和鲁棒性。在趋势预测中，随机森林能够捕捉数据中的复杂模式和非线性关系。\n' +
    '# 特点\n' +
    '### 1. 高准确性：随机森林通过集成多个决策树的预测结果，降低了模型的方差，提高了预测的准确性。\n' +
    '### 2. 自动特征选择：随机森林在构建决策树的过程中，可以评估特征的重要性，从而实现自动特征选择。\n' +
    '### 3. 强大的非线性拟合能力：随机森林能够处理高度复杂的数据模式，适用于非线性趋势的预测。\n' +
    '### 4. 鲁棒性：由于集成了多个决策树，随机森林对异常值和噪声具有较强的鲁棒性。\n' +
    '### 5. 易于实现和并行化：随机森林模型易于实现，并且其训练过程可以并行化，提高了计算效率。\n' +
    '### 6. 多变量处理能力：随机森林能够同时处理多个变量，捕捉它们与预测目标之间的复杂关系。\n',
  'SVM的趋势预测': '# 支持向量机（SVM）趋势预测\n' +
    '## 支持向量机是一种监督学习模型，用于分类和回归任务。在趋势预测中，SVM通过找到数据中的最优边界或超平面，对数据的未来趋势进行预测。\n' +
    '# 特点\n' +
    '### 1. 优秀的泛化能力：SVM通过选择支持向量来构建模型，这些向量定义了决策边界，使得模型具有很好的泛化能力。\n' +
    '### 2. 核技巧：SVM使用核函数将数据映射到高维空间，以处理非线性趋势预测问题。\n' +
    '### 3. 正则化控制：通过正则化参数控制模型的复杂度，避免过拟合，确保模型的稳定性。\n' +
    '### 4. 适用于小样本数据：SVM在小样本情况下也能表现出较好的预测性能，适合样本量不足的趋势预测任务。\n' +
    '### 5. 模型解释性：相比于一些黑盒模型，SVM具有一定的解释性，特别是通过支持向量可以了解模型决策的关键数据点。\n' +
    '### 6. 多类趋势预测：通过适当的策略，SVM可以扩展到多类趋势预测问题。\n',
  'GRU的趋势预测': '# 门控循环单元（GRU）趋势预测\n' +
    '## 门控循环单元是一种特殊类型的递归神经网络（RNN），设计用于处理序列数据，能够捕捉时间序列中的动态特征和长期依赖关系。在趋势预测中，GRU能够学习数据随时间变化的模式，并预测未来的趋势。\n' +
    '# 特点\n' +
    '### 1. 捕捉时间序列特性：GRU通过其门控机制能够捕捉时间序列数据中的短期和长期依赖关系，适用于具有时间连贯性的趋势预测。\n' +
    '### 2. 门控调节信息流：利用更新门和重置门，GRU能够有选择地更新或保留状态信息，从而适应不同时间尺度的趋势变化。\n' +
    '### 3. 处理非线性动态：GRU能够处理复杂的非线性时间序列数据，对于预测不规则或周期性变化的趋势特别有效。\n' +
    '### 4. 避免梯度消失问题：GRU的设计有助于缓解传统RNN中的梯度消失问题，使得网络能够学习长期时间依赖。\n' +
    '### 5. 易于集成和训练：现代深度学习框架提供了GRU的实现，易于集成到趋势预测模型中，并支持大规模数据集的训练。\n' +
    '### 6. 多步时间预测：GRU不仅可以进行单步预测，还可以通过序列到序列模型进行多步时间预测。\n',
  'LSTM的趋势预测': '# 长短期记忆网络（LSTM）在趋势预测中的应用\n' +
    '## 长短期记忆网络是一种高级的递归神经网络（RNN），专为解决传统RNN在处理长序列数据时遇到的梯度消失或梯度爆炸问题而设计。LSTM在趋势预测中能够学习时间序列数据中的长期依赖关系，并进行有效的预测。\n' +
    '# 特点\n' +
    '### 1. 长期依赖学习：LSTM的特别设计使其能够捕捉时间序列中相隔很远的依赖关系，这对于理解长期趋势至关重要。\n' +
    '### 2. 门控机制：过其复杂的门控机制（输入门、遗忘门、输出门），LSTM能够决定信息的保留和遗忘，从而适应时间序列数据的变化。\n' +
    '### 3. 处理非线性：LSTM能够处理时间序列数据中的非线性模式，适用于预测复杂的趋势变化。\n' +
    '### 4. 避免过拟合：由于其门控单元的结构，LSTM在训练过程中更不容易过拟合，提高了模型的泛化能力。\n' +
    '### 5. 多步预测能力：LSTM可以设计为序列到序列模型，进行多步趋势预测，而不仅仅是单步预测。\n',
  '添加噪声': '### 添加噪声算法可对输入信号添加如高斯噪声等常见的噪声\n***\n ### 添加噪声的主要应用有：\n **1. 增强信号检测。在某些特定的非线性系统中，噪声的存在能够增强微弱信号的检测能力，这种现象被称为随机共振。**\n \
  **2. 减小重构误差。如果在信号处理中只加入正的白噪声，那么在重构过程中可能会多出来加入的白噪声，从而增大了重构误差。因此，加入正负序列的白噪声可以在分解过程中相互抵消，减小重构误差。**\n### 本模块包含算法如下:\n **高斯白噪声： 高斯白噪声(White Gaussian Noise)在通信、信号处理及科学研究等\
  多个领域中扮演着重要角色。它作为一种理想的噪声模型，具有独特的统计特性和功率谱分布，为系统性能评估、算法测试及信号分析提供了有力工具**',
  '插值处理': '### 插值处理算法可以对输入信号进行插值操作\n***\n ### 插值处理的主要应用有：\n **1. 数值计算。函数逼近：插值方法，如拉格朗日插值多项式，可以用于在给定节点上逼近任意函数，从而在节点外的位置计算函数的近似值。**\n \
  **2. 图像处理。图像放大与缩小：双线性插值等方法可以用于图像的放大或缩小，以获取更高分辨率或适当尺寸的图像。图像平滑：线性插值等方法可以用于平滑图像中的噪声或异常值，提高图像质量。**\
  **3. 数据净化。缺失值处理：插值法是一种有效的缺失值处理方法，能够根据不同情况选择合适的插值类型进行估算，实现数据的完整性和连续性。数据平滑：插值法可以用于构建连续且平滑的函数模型来拟合数据，消除噪声和异常值的影响，提高数据质量。**\n \
  ### 本插值处理模块中包含算法：\n **多项式插值算法、双三次插值算法、拉格朗日插值算法、牛顿插值算法、线性插值算法**',
  'TDF': '### 特征提取算法可对输入信号进行人工特征提取\n***\n ### 特征提取算法中主要包括信号时域特征和频域特征的提取：\n **1. 时域特征。定义：时域特征描述的是信号随时间变化的关系。在时域中，信号被表示为时间的函数，其动态信号描述信号在不同时刻的取值。\
  特点：时域表示较为形象与直观，对于正弦信号等简单波形的时域表达，可以通过幅值、频率、相位三个基本特征值唯一确定。时域分析能够直接反映信号随时间的实时变化。**\n **2. 频域特征。定义：频域特征描述的是信号在频率方面的特性。\
  频域分析是通过对信号进行傅立叶变换等数学方法，将信号从时间域转换到频率域，从而研究信号的频率结构。特点：频域分析能够深入剖析信号的本质，揭示信号中不易被时域分析发现的特征。频域特征通常用于表达信号的周期性信息。**\n ### 本特征提取算法中提取的时域和频域的特征包括：\
  \n **1. 时域特征：均值、方差、标准差、峰度、偏度、四阶累积量、六阶累积量、最大值、最小值、中位数、峰峰值、整流平均值、均方根、方根幅度、波形因子、峰值因子、脉冲因子、裕度因子**\n \
  **1. 频域特征：重心频率、均方频率、均方根频率、频率方差、频率标准差、谱峭度的均值、谱峭度的标准差、谱峭度的峰度、谱峭度的偏度**',
  '层次分析模糊综合评估': '### 层次分析模糊综合评估法是一种将模糊综合评价法和层次分析法相结合的评价方法\n***\n ### 算法优点：\n **1. 可以综合考虑多个因素的影响，给出全面评价结果。**\n \
  **2. 评价结果是一个矢量，而不是一个点值，包含的信息比较丰富，既可以比较准确的刻画被评价对象，又可以进一步加工，得到参考信息。**\n **3. 模糊评价通过精确的数字手段处理模糊的评价对象，能对蕴藏信息呈现模糊性的资料作出比较科学、合理、贴近实际的量化评价。**'
}

const introduction_to_show = ref('# 你好世界')  // 需要展示在可视化建模区的算法介绍
const showPlainIntroduction = ref(false)

// 点击标签页切换单传感器和多传感器算法
const handleClick = (tab, event) => {
  console.log(tab, event)
}

// 算法介绍，点击算法选择区内的具体算法，将其算法介绍展示在可视化建模区
const showIntroduction = (algorithm) => {
  results_view_clear()
  showStatusMessage.value = false
  showPlainIntroduction.value = true
  introduction_to_show.value = plain_introduction[algorithm]

}

// 算法选择菜单下拉展示
const menu_details_second = ref({

})

const menu_details_third = ref({

})

// 特征提取所选择的特征
// const features = ref([])
const features = ref(['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
  '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度'])

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
    let nodeList = []
    while (current) {
      nodeList.push(current.value)
      current = current.next;
    }
    return nodeList
  }
  length() {
    if (this.head) {
      let len = 1
      let p = this.head.next
      while (p) {
        p = p.next
        len += 1
      }
      return len
    }
    return 0
  }
  search(target_value) {
    if (this.head == null) {
      return false
    } else {
      let current = this.head
      while (current) {
        if (current.value == target_value) {
          return current
        }
        current = current.next
      }
      return false
    }
  }
}

const logout = () => {
  router.push('/')
  console.log('用户退出登录')
}

// 标签与节点id的转换
const display_label_to_id = (display_label) => {
  nodeList.value.forEach(node => {
    if (node.display_label == display_label) {
      return node.id
    }
  })
}
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
  let node_list = nodeList.value.slice()
  let nodeId_to_find = 0
  node_list.forEach(node => {
    if (node.label == label) {
      nodeId_to_find = node.id
    }
  })
  return nodeId_to_find
}

// 连线操作
const linkedList = new LinkedList()
onMounted(() => {
  username.value = window.localStorage.getItem('username') || '用户名未设置'
  console.log('username: ', username.value)

  document.querySelector('.el-main').classList.add('has-background');
  plumbIns = jsPlumb.getInstance()
  jsPlumbInit()

  plumbIns.bind("connection", function (info) {
    let sourceId = info.connection.sourceId
    let targetId = info.connection.targetId

    let id_to_label = {}

    nodeList.value.forEach(node => {
      let id = node.id
      let label = node.label
      id_to_label[id] = label
    })
    if (linkedList.head == null) {
      linkedList.append(id_to_label[sourceId])
      linkedList.append(id_to_label[targetId])
    } else {
      if (linkedList.head.value == id_to_label[targetId]) {
        linkedList.insertAtHead(id_to_label[sourceId])
      } else {
        linkedList.append(id_to_label[targetId])
      }
    }


    // 除去在linkedList中的节点，其他节点不能作为连线操作的出发点
    let linked = linkedList.get_all_nodes()
    // for(let [value, key] of id_to_label){
    //   if (linked.indexOf(key) == -1){
    //     plumbIns
    //   }
    // }

    console.log('linked: ' + linked)


  })

  plumbIns.bind('beforeConnect', function (info) {
    console.log('调用')
    let sourceId = info.connection.sourceId
    let targetId = info.connection.targetId
    if (sourceId == '3.1') {
      return false
    }
  })
})

const deff = {
  jsplumbSetting: {
    // 动态锚点、位置自适应
    Anchors: ['Right', 'Left'],
    anchor: ['Right', 'Left'],
    // 容器ID
    Container: 'efContainer',
    // 连线的样式，直线或者曲线等，可选值:  StateMachine、Flowchart，Bezier、Straight
    // Connector: ['Bezier', {curviness: 100}],
    // Connector: ['Straight', { stub: 20, gap: 1 }],
    Connector: ['Flowchart', { stub: 30, gap: 1, alwaysRespectStubs: false, midpoint: 0.5, cornerRadius: 10 }],
    // Connector: ['StateMachine', {margin: 5, curviness: 10, proximityLimit: 80}],
    // 鼠标不能拖动删除线
    ConnectionsDetachable: false,
    // 删除线的时候节点不删除
    DeleteEndpointsOnDetach: false,
    /**
     * 连线的两端端点类型：圆形
     * radius: 圆的半径，越大圆越大
     */
    Endpoint: ['Dot', { radius: 10, cssClass: 'ef-dot', hoverClass: 'ef-dot-hover' }],
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
    EndpointStyle: { fill: '#1879ffa1', outlineWidth: 3, },
    // 是否打开jsPlumb的内部日志记录
    LogEnabled: true,
    /**
     * 连线的样式
     */
    PaintStyle: {
      // 线的颜色
      stroke: '#4CAF50',
      // 线的粗细，值越大线越粗
      strokeWidth: 7,
      // 设置外边线的颜色，默认设置透明，这样别人就看不见了，点击线的时候可以不用精确点击，参考 https://blog.csdn.net/roymno2/article/details/72717101
      outlineStroke: 'transparent',
      // 线外边的宽，值越大，线的点击范围越大
      outlineWidth: 5,
    },
    DragOptions: { cursor: 'pointer', zIndex: 2000 },
    ConnectionOverlays: [
      ['Custom', {
        create() {
          const el = document.createElement('div')
          // el.innerHTML = '<select id=\'myDropDown\'><option value=\'foo\'>foo</option><option value=\'bar\'>bar</option></select>'
          return el
        },
        location: 0.7,
        id: 'customOverlay',
      }],
    ],
    /**
     *  叠加 参考： https://www.jianshu.com/p/d9e9918fd928
     */
    Overlays: [
      // 箭头叠加
      ['Arrow', {
        width: 25, // 箭头尾部的宽度
        length: 8, // 从箭头的尾部到头部的距离
        location: 1, // 位置，建议使用0～1之间
        direction: 1, // 方向，默认值为1（表示向前），可选-1（表示向后）
        foldback: 0.623, // 折回，也就是尾翼的角度，默认0.623，当为1时，为正三角
      }],
      // ['Diamond', {        //     events: {        //         dblclick: function (diamondOverlay, originalEvent) {        //             console.log('double click on diamond overlay for : ' + diamondOverlay.component)        //         }        //     }        // }],
      ['Label', { label: '', location: 0.1, cssClass: 'aLabel', }],

    ],
    // 绘制图的模式 svg、canvas
    RenderMode: 'canvas',
    // 鼠标滑过线的样式
    HoverPaintStyle: { stroke: 'red', strokeWidth: 10 },
    // 滑过锚点效果
    EndpointHoverStyle: { fill: 'red' },
    Scope: 'jsPlumb_DefaultScope', // 范围，具有相同scope的点才可连接
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
      cssClass: 'flowLabel',
    },
  },
  /**
   * 源点配置参数
   */
  jsplumbSourceOptions: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: '.node-drag',
    filterExclude: false,
    anchor: ['Continuous', { faces: ['right'] }],
    // 是否允许自己连接自己
    allowLoopback: false,
    maxConnections: -1,
  },
  // 参考 https://www.cnblogs.com/mq0036/p/7942139.html
  jsplumbSourceOptions2: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: '.node-drag',
    filterExclude: false,
    // anchor: 'Continuous',
    // 是否允许自己连接自己
    allowLoopback: true,
    connector: ['Flowchart', { curviness: 50 }],
    connectorStyle: {
      // 线的颜色
      stroke: 'red',
      // 线的粗细，值越大线越粗
      strokeWidth: 1,
      // 设置外边线的颜色，默认设置透明，这样别人就看不见了，点击线的时候可以不用精确点击，参考 https://blog.csdn.net/roymno2/article/details/72717101
      outlineStroke: 'transparent',
      // 线外边的宽，值越大，线的点击范围越大
      outlineWidth: 10,
    },
    connectorHoverStyle: { stroke: 'red', strokeWidth: 2 },
  },
  jsplumbTargetOptions: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: '.node-drag',
    filterExclude: false,
    // 是否允许自己连接自己
    anchor: ['Continuous', { faces: ['left'] }],
    allowLoopback: false,
    dropOptions: { hoverClass: 'ef-drop-hover' },
  },
}
const done = ref(false)
const dialogmodle = ref(false)

let model_check_right = false
// 检查模型
const check_model = () => {
  console.log(linkedList.get_all_nodes())
  let id_to_module = {}
  let algorithms = []
  let algorithm_schedule = []
  let module_schedule = []
  if (nodeList.value.length == 1) {
    module_schedule.push(nodeList.value[0].label)
    algorithm_schedule.push(nodeList.value[0].label_display)
  } else {
    // module_schedule = linkedList.get_all_nodes()
    let all_connections = plumbIns.getConnections();
    console.log('all_connections: ', all_connections)
    // 获取连线元素的单向映射
    let connectionsMap: any = {};  
    all_connections.forEach(connection => {  
      const sourceId = connection.sourceId; 
      const targetId = connection.targetId;  
    
      // 如果源元素ID不在connectionsMap中，则初始化为空数组  
      if (!connectionsMap[sourceId]) {  
          connectionsMap[sourceId] = [];  
      }
      connectionsMap[sourceId].push(targetId);  
    })
    // console.log('connectionsMap: ', connectionsMap)
    
    // 寻找逻辑上的第一个元素
    function findStartElement(connectionsMap: any) {  
      // 创建一个集合来存储所有元素的ID  
      const allElements = new Set(Object.keys(connectionsMap).concat(...Object.values(connectionsMap).map(list => list)));  
    
      // 遍历所有元素，查找没有入度的元素  
      for (const elementId of allElements) {  
          let hasIncomingConnection = false;  
          for (const connections of Object.values(connectionsMap)) {  
              if (connections.includes(elementId)) {  
                  hasIncomingConnection = true;  
                  break;  
              }  
          }  
          if (!hasIncomingConnection) {  
              return elementId; // 找到没有入度的元素，即起点  
          }  
      }  
    
      // 如果没有找到没有入度的元素，则可能图不是线性的，或者connectionsMap构建有误  
      throw new Error("No start element found. The graph may not be linear or the connectionsMap may be incorrect.");  
    }
    let startElementId = findStartElement(connectionsMap);
    
    // 寻找逻辑上的下一个元素
    function findNextElementIdInSequence(currentElementId, connectionsMap) {  
      
      const connections = connectionsMap[currentElementId];  
    
      // 假设序列是线性的 
      if (connections && connections.length > 0) {  
          return connections[0]; // 返回序列中的下一个元素ID  
      }  
      return null;  
    }
    
    function traverseLinearSequence(startElementId, connectionsMap, visited = new Set(), order = []) {  
      // 检查是否已访问过当前元素  
      if (visited.has(startElementId)) {  
          return;  
      }  
    
      visited.add(startElementId); // 标记为已访问  
      order.push(startElementId); // 将元素添加到顺序数组中  
     
      let nextElementId = findNextElementIdInSequence(startElementId, connectionsMap);  
    
      if (nextElementId !== null) {  
          // 递归遍历下一个元素  
          traverseLinearSequence(nextElementId, connectionsMap, visited, order);  
      }  
    
      return order;  
    }
    let sequenceOrder = traverseLinearSequence(startElementId, connectionsMap);  
    console.log('sequenceOrder: ', sequenceOrder);
    nodeList.value.forEach(node => {
      let id = node.id
      let label = node.label
      let algorithm = node.label_display
      id_to_module[id] = label
      algorithms.push(algorithm)
    })
    
    sequenceOrder.forEach(id => {
      module_schedule.push(id_to_module[id])
    });
    

    for (let i = 0; i < module_schedule.length; i++) {
      let module = module_schedule[i]
      nodeList.value.forEach(node => {
        if (node.label == module) {
          algorithm_schedule.push(node.label_display)
        }
      });
    }
  }
  

  let module_str = Object.values(module_schedule).join('')
  let algorithm_str = Object.values(algorithm_schedule).join('')
  console.log('module_str: ' + module_str)
  console.log('algorithm_str: ' + algorithm_str)
  // 判断子串后是否有更多的文本
  const moreText = (text, substring) => {
    const position = text.indexOf(substring);
    if (position === -1) {
      return false;
    }
    const endPosition = position + substring.length;
    return endPosition < text.length;
  }

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
  }
  if (nodeList.value.length) {
    if (nodeList.value.length == 1) {
      if (!module_str.match('插值处理') && !module_str.match('特征提取') && !algorithm_str.match('GRU的故障诊断') && !algorithm_str.match('LSTM的故障诊断') && !algorithm_str.match('小波变换降噪') && !module_str.match('无量纲化')) {
        ElMessage({
          message: '该算法无法单独使用，请结合相应的算法',
          type: 'warning'
        })
        return
      } else {
        // 进行模型参数设置的检查
        let check_params_right = check_model_params()
        if (check_params_right) {
          ElMessage({
            showClose: true,
            message: '模型正常，可以保存并运行',
            type: 'success'
          })
          model_check_right = true
          updateStatus('模型建立并已通过模型检查')
        } else {
          ElMessage({
            showClose: true,
            message: '模型参数未设置',
            type: 'warning'
          })
          return
        }
      }
    } else {
      if (linkedList.length() != nodeList.value.length) {
        ElMessage({
          message: '请确保图中所有模块均已建立连接，且没有多余的模块',
          type: 'warning'
        })
        return
      } else {
        if (module_str.match('特征选择故障诊断') && !module_str.match('特征提取特征选择故障诊断')) {
          ElMessage({
            showClose: true,
            message: '因模型中包含故障诊断，建议在特征选择之前包含特征提取',
            type: 'warning'
          })
          return
        } else if (module_str.match('特征提取故障诊断')) {
          let source_id = label_to_id('特征提取')
          let current = linkedList.search('特征提取')
          let next = current.next.value
          let target_id = label_to_id(next)

          let connection = plumbIns.getConnections({ source: source_id, traget: target_id })
          console.log('connection: ', connection)

          plumbIns.select({ source: source_id, target: target_id }).setPaintStyle({
            stroke: '#E53935',
            strokeWidth: 7,
            outlineStroke: 'transparent',
            outlineWidth: 5,

          });
          ElMessage({
            showClose: true,
            message: '因模型中包含故障诊断，建议在特征提取之后包含特征选择',
            type: 'warning'
          })
          return
        } else if (module_str.match('层次分析模糊综合评估') && !module_str.match('特征提取')) {
          ElMessage({
            showClose: true,
            message: '因模型中包含层次分析模糊综合评估，建议在此之前包含特征提取',
            type: 'warning'
          })
          return
        } else if (module_str.match('层次分析模糊综合评估') && (module_str.match('LSTM的故障诊断') || module_str.match('SVM的故障诊断'))) {
          ElMessage({
            showClose: true,
            message: '使用深度学习模型的故障诊断无法为健康评估提供有效的评估依据，建议使用机器学习的故障诊断配合健康评估！',
            type: 'warning'
          })
          return
        } else if (module_str.match('层次分析模糊综合评估') && moreText(module_str, '层次分析模糊综合评估')) {
          ElMessage({
            showClose: true,
            message: '注意健康评估之后无法连接更多的模块',
            type: 'warning'
          })
          return
        } else if (algorithm_str.match('多传感器') && algorithm_str.match('单传感器')) {
          ElMessage({
            showClose: true,
            message: '针对单传感器的算法无法与针对多传感器的算法共用',
            type: 'warning'
          })
          return
        } else if (module_str.match('故障诊断')) {
          if (moreText(module_str, '故障诊断')) {
            if (!checkSubstrings(module_str, '故障诊断', ['层次分析模糊综合评估', '趋势预测'])) {
              ElMessage({
                showClose: true,
                message: '注意故障诊断之后仅能进行趋势预测或是健康评估！',
                type: 'warning'
              })
              let source_id = label_to_id('故障诊断')
              let current = linkedList.search('故障诊断')
              let next = current.next.value
              let target_id = label_to_id(next)
              
              let connection = plumbIns.getConnections({ source: source_id, traget: target_id })
              console.log('connection: ', connection)

              plumbIns.select({ source: source_id, target: target_id }).setPaintStyle({
                stroke: '#E53935',
                strokeWidth: 7,
                outlineStroke: 'transparent',
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
              return
            }
          }
          if (algorithm_str.match('SVM的故障诊断')) {
            if (!module_str.match('无量纲化') || !checkSubstrings(module_str, '无量纲化', ['故障诊断'])) {
              ElMessage({
                showClose: true,
                message: '因模型中包含SVM的故障诊断，需要在此之前加入标准化操作',
                type: 'warning'
              })
              return
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
        let check_params_right = check_model_params()
        if (check_params_right) {
          ElMessage({
            showClose: true,
            message: '模型正常，可以保存并运行',
            type: 'success'
          })
          model_check_right = true
          updateStatus('模型建立并已通过模型检查')
        } else {
          ElMessage({
            showClose: true,
            message: '模型参数未设置',
            type: 'warning'
          })
          return
        }
      }
    }
  } else {
    ElMessage({
      message: '请先建立模型',
      type: 'warning'
    })
    return
  }
  canSaveModel.value = false
  // canStartProcess.value = false
}

// 进度条完成度
let processing = ref(false)
let percentage = ref(0)
// let timerId = null
let fastTimerId = null; // 快速定时器ID  
let slowTimerId = null; // 慢速定时器ID  

let response_results = {}


const username = ref('')

// 创建一个CancelToken源  
let cancel;  
  
const source = axios.CancelToken.source();  
cancel = source.cancel; // 暴露cancel函数  


//上传文件，点击开始
const run = () => {

  // 发送文件到后端
  // if (!file_list.value.length) {
  //   ElNotification({
  //     title: 'WARNING',
  //     message: '数据文件不能为空,请先上传数据',
  //     type: 'warning',
  //   })
  //   return
  // } else {
  //   const filename = file_list.value[0].name
  //   const filetype = filename.substring(filename.lastIndexOf('.'))
  //   if (filetype != '.xlsx' && filetype != '.npy' && filetype != '.wav' && filetype != '.audio' && filetype != '.csv' && filetype != '.mat') {
  //     ElNotification({
  //       title: 'WARNING',
  //       message: '文件类型只能为xlsx、npy、wav、audio、csv、mat',
  //       type: 'warning',
  //     })
  //     return
  //   }
  // }


  // let datafile = file_list.value[0].raw
  if (!using_datafile.value){
    ElMessage({
      message: '请先加载数据',
      type: 'warning'
    })
    return
  }

  const data = new FormData()
  console.log('datafile: ', using_datafile.value)
  data.append("file_name", using_datafile.value)
  data.append('params', JSON.stringify(content_json))
  ElNotification.info({
    title: 'SUCCESS',
    message: '正在处理，请等待'
  })
  canShutdown.value = false

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
  processing.value = true

  // console.log("",data)
  api.post('user/run_with_datafile_on_cloud/', data,
    {
      headers: { "Content-Type": 'multipart/form-data' },
      cancelToken: source.token, // 将cancelToken传递给axios  
    }
  ).then((response) => {
    console.log('response: ', response)
    console.log('response.status: ', response.status)
    if (response.status === 200) {
      if (!process_is_shutdown.value) {
        ElNotification.success({
          title: 'SUCCESS',
          message: '处理完成',
        })
        response_results = response.data.results
        console.log('resoonse_results: ', response_results)
        missionComplete.value = true
        // setTimeout(function () { processing.value = false; percentage.value = 50 }, 500)
        // percentage.value = 100
        // clearInterval(timerId);
        clearInterval(fastTimerId);
        clearInterval(slowTimerId);
        setTimeout(function () { processing.value = false }, 700)
        percentage.value = 100;
        canShutdown.value = true
        status_message_to_show.value = status_message.success
        results_view_clear()

        showStatusMessage.value = true
        showPlainIntroduction.value = false
      } else {
        process_is_shutdown.value = false
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
    .catch(error => {
      // if (axios.isCancel(error)) {
      //   console.log('Request canceled', error.message);
      //   return
      // }
      if (error.response) {
        // 请求已发出，服务器响应了状态码，但不在2xx范围内  
        console.log(error.response.status); // HTTP状态码  
        console.log(error.response.statusText); // 状态消息  

      } else if (error.request) {
        // 请求已发起，但没有收到响应  
        console.log(error.request);
      } else {
        // 设置请求时触发了错误  
        console.error('Error', error.message);
      }

      ElNotification.error({
        title: 'ERROR',
        message: '处理失败，请重试',
      })
      loading.value = false
      processing.value = false

    })
}


const process_is_shutdown = ref(false)

// 终止进程
const shutdown = () => {
  axios.request(
    {
      method: 'GET',
      url: 'http://127.0.0.1:8000/shut'
    }
  ).then((response) => {
    if (response.data.status == 'shutdown' && processing.value == true) {
      loading.value = false
      processing.value = false
      missionComplete.value = false
      ElNotification.info({
        title: 'INFO',
        message: '进程已终止'
      })
      process_is_shutdown.value = true
      status_message_to_show.value = status_message.shutdown
      showStatusMessage.value = true
      // cancel('Operation canceled by the user.');  
    }
  }).catch(function (error) {
    // 处理错误情况  
    ElNotification.error({
      title: 'ERROR',
      message: '请求终端进程失败'
    })
    console.log('请求中断进程失败：' + error)
  });
}

const isshow = ref(false)
const selects = ref(false)
const nodeList = ref([])
const efContainerRef = ref()
const nodeRef = ref([])
const content_json = {
  'modules': [],
  'algorithms': {},
  'parameters': {},
  'schedule': []
}

let plumbIns
let missionComplete = ref(false)
let loading = ref(false)
let modelSetup = ref(false)
const handleclear = () => {
  done.value = false
  nodeList.value = []  // 可视化建模区的节点列表
  // features.value = []  // 特征提取选择的特征
  features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
    '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']
  json_file_clear()    // 向后端发送的模型信息
  isshow.value = false
  plumbIns.deleteEveryConnection()
  plumbIns.deleteEveryEndpoint()
  linkedList.head = null
  linkedList.tail = null
  missionComplete.value = false // 程序处理完成
  modelSetup.value = false   // 模型设置完成
  showPlainIntroduction.value = false
  showStatusMessage.value = false
  model_has_been_saved = false //复用历史模型，不做模型检查
  to_rectify_model.value = false  // 禁用修改模型
  canCompleteModeling.value = true
  canCheckModel.value = true
  canSaveModel.value = true
  canStartProcess.value = true
  updateStatus('未建立模型')

  results_view_clear()
}
const json_file_clear = () => {
  content_json.modules = []
  content_json.algorithms = {}
  content_json.parameters = {}
  content_json.schedule = []
}
const jsPlumbInit = () => {
  plumbIns.importDefaults(deff.jsplumbSetting)
}

//处理拖拽，初始化节点的可连接状态
const handleDragend = (ev, algorithm, node) => {

  // 拖拽进来相对于地址栏偏移量
  const evClientX = ev.clientX
  const evClientY = ev.clientY
  let left = evClientX - 300 + 'px'
  let top = 50 + 'px'
  const nodeId = node.id
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
    parameters: node.parameters
  }
  // 特征提取默认参数
  // if (nodeInfo.label_display.indexOf('时域特征') > -1){
  //   features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
  //     '波形因子', '峰值因子', '脉冲因子', '裕度因子']

  // }else if (nodeInfo.label_display.indexOf('频域特征') > -1){
  //   features.value = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度']
  // }else if (nodeInfo.label_display.indexOf('时域和频域') > -1){
  //   features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
  //     '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度']
  // }
  if (nodeInfo.label_display.indexOf('时域和频域') > -1) {
    features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
      '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度']
  }else {
    if (nodeInfo.label_display.indexOf('时域特征') > -1){
      features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
        '波形因子', '峰值因子', '脉冲因子', '裕度因子']
    }else if (nodeInfo.label_display.indexOf('频域特征') > -1){
      features.value = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度']
    }
  }
  console.log(nodeInfo)
  //算法模块不允许重复
  if (nodeList.value.length === 0) {
    nodeList.value.push(nodeInfo)
  } else {
    let isDuplicate = false;
    for (let i = 0; i < nodeList.value.length; i++) {
      let nod = nodeList.value[i];
      if (nod.id == node.id) {
        // window.alert('不允许出现重复模块');
        ElMessage({
          message: '不允许出现同一类别的算法',
          type: 'warning'
        })
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
    plumbIns.draggable(nodeId, { containment: "efContainer" })
    // if (node.id > 2) {
    //   plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)
    //   return
    // }
    if (node.id < 4) {
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions)
    }

    // plumbIns.addEndpoint(nodeId, deff.jsplumbTargetOptions)

    plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)

  })
}

// 删除节点的操作
const deleteNode = (nodeId) => {
  if (!modelSetup.value) {
    nodeList.value = nodeList.value.filter(node => node.id !== nodeId);
    plumbIns.deleteEveryConnection()
    plumbIns.deleteEveryEndpoint()
    linkedList.head = null
    linkedList.tail = null
    canCheckModel.value = true
    canStartProcess.value = true
    canShutdown.value = true
    canSaveModel.value = true
  }
}


const handleMouseup = (ev, data) => { // 在图表中拖拽节点时，设置他的新的位置
  // nodeList.value.forEach(node => {
  //   if (node.id === data.id) {
  //     data.nodeContainerStyle.left = ev.clientX
  //     data.nodeContainerStyle.top = ev.clientY
  //     // node.value.nodeContainerStyle.left = ev.clientX + 'px'
  //     // node.value.nodeContainerStyle.top = ev.clientY + 'px'
  //   }
  // })
  if (!done.value) {
    length = nodeList.value.length
    for (let i = 0; i < length; i++) {
      let node = nodeList.value[i]
      if (node.id === data.id) {
        // setTimeout(()=>{
        //   data.nodeContainerStyle.left = ev.clientX - 290 
        //   data.nodeContainerStyle.top = ev.clientY - 80 
        // }, 2)
        // data.nodeContainerStyle.left = ev.clientX - 290 
        // data.nodeContainerStyle.top = ev.clientY - 80 

        // node.value.nodeContainerStyle.left = ev.clientX + 'px'
        // node.value.nodeContainerStyle.top = ev.clientY + 'px'
        nodeList.value[i].nodeContainerStyle.left = ev.clientX - 300 + 'px'
        nodeList.value[i].nodeContainerStyle.top = ev.clientY - 100 + 'px'
      }
    }
    console.log('拖动节点，目前各节点信息为：', nodeList.value)
  }

}



const modelsetting = () => {
  selects.value = !selects.value
}

const dialogFormVisible = ref(false)  // 控制对话框弹出，输入要保存的模型的名称

// 提交的模型相关信息
const model_info_form = ref({
  name: '',
  region: '',
})

// 检查模型参数设置
const check_model_params = () => {
  for (let i = 0; i < nodeList.value.length; i++) {
    let dict = nodeList.value[i]

    if (!dict.use_algorithm) {
      return false
    }

    if (dict.id == '1.2') {
      if (!features.value.length) {
        return false
      }
    }
  }

  return true
}

//保存模型并取消拖拽动作                 
const saveModelSetting = (saveModel, schedule) => {

  done.value = true

  // dialogFormVisible.value = true
  // selects.value = !selects.value
  json_file_clear()
  for (let i = 0; i < nodeList.value.length; i++) {
    let dict = nodeList.value[i]

    if (!dict.use_algorithm) {
      ElMessage({
        message: '请设置每个算法的必选属性',
        type: 'error'
      })
      console.log('dict.use_algorithm is empty! return')
      return
    }

    content_json.algorithms[dict.label] = dict.use_algorithm
    if (!content_json.modules.includes(dict.label)) {
      content_json.modules.push(dict.label);
    }

    // 选择特征提取需要展示的参数
    if (dict.id == '1.2') {
      let params = dict.parameters[dict.use_algorithm]
      if (!features.value.length) {
        ElMessage({
          message: '请设置每个算法的必选属性',
          type: 'error'
        })
        return
      }
      features.value.forEach(element => {
        if (params[element] == false) {
          params[element] = true
        }
      });
      content_json.parameters[dict.use_algorithm] = params
      continue
    }
    content_json.parameters[dict.use_algorithm] = dict.parameters[dict.use_algorithm]
    // console.log(dict.use_algorithm + '\'s params are: ' + dict.parameters[dict.use_algorithm])

  }
  if (!model_check_right && saveModel) {
    ElMessage({
      message: '请先建立模型并通过模型检查！',
      type: 'warning'
    })
    return
  }
  let current = linkedList.head;
  content_json.schedule.length = 0
  console.log('nodeList: ', nodeList.value)
  // 如果只有一个节点，无需建立流程
  if (nodeList.value.length == 1) {
    content_json.schedule.push(nodeList.value[0].label)
    // console.log('只有一个节点，无需建立流程')
  } else {
    if (!saveModel) {
      console.log('schedule: ', schedule)
      content_json.schedule = schedule
      console.log('content_json: ', content_json)
    } else {
      if (!current) {
        ElNotification({
          title: 'WARNING',
          message: '未建立流程，请先建立流程',
          type: 'warning',
        })
        return
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
  dialogmodle.value = saveModel
}

// 完成模型名称等信息的填写后，确定保存模型
const save_model_confirm = () => {
  let data = new FormData()
  data.append('model_name', model_info_form.value.name)
  let nodelist_length = nodeList.value.length
  let nodelist_info = nodeList.value
  // for (let i = 0; i < nodelist_length; i++){
  //   nodelist_info[i].nodeContainerStyle.left += 'px'
  //   nodelist_info[i].nodeContainerStyle.top += 'px'
  // }

  let model_info = { "nodeList": nodelist_info, "connection": content_json.schedule }
  data.append('model_info', JSON.stringify(model_info))
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
  api.post('http://127.0.0.1:8000/user_save_model/', data,
    {
      headers: { "Content-Type": 'multipart/form-data' }
    }
  ).then((response) => {
    if (response.data.message == 'save model success') {
      ElMessage({
        message: '保存模型成功',
        type: 'success'
      })
      fetch_models()
      models_drawer.value = false       // 关闭历史模型抽屉
      dialogFormVisible.value = false    // 关闭提示窗口
      dialogmodle.value = false
      canStartProcess.value = false     // 保存模型成功可以运行
      modelSetup.value = true                 // 模型保存完成
      updateStatus('当前模型已保存')
    } else if(response.data.status == 400) {
      ElMessage({
        message: '已有同名模型，保存模型失败',
        type: 'error'
      })
    }
  }).catch(error=>{
    ElMessage({
      message: '保存模型请求失败',
      type: 'error'
    })
    console.log('save model error: ', error)
  })
}
// let showMenu = ref(false)
// const handleRightClick = (event) => {
//   event.preventDefault(); // 阻止默认的右键菜单
//   showMenu.value = true; // 显示自定义的下拉菜单
//   console.log('右击')
//   console.log(event.target.getAttribute('id'))
// }
const show1 = ref(false)

// 结果可视化区域显示（无后端响应时间限制）
const result_show = ref(false)

// 健康评估结果展示
const health_evaluation = ref('')
const display_health_evaluation = ref(false)
const activeName1 = ref('first')
const health_evaluation_figure_1 = ref('data:image/png;base64,')
const health_evaluation_figure_2 = ref('data:image/png;base64,')
const health_evaluation_figure_3 = ref('data:image/png;base64,')

const health_evaluation_display = (results_object) => {

  display_health_evaluation.value = true
  let figure1 = results_object.层级有效指标_Base64
  let figure2 = results_object.二级指标权重柱状图_Base64
  let figure3 = results_object.评估结果柱状图_Base64

  health_evaluation.value = results_object.评估建议
  health_evaluation_figure_1.value = 'data:image/png;base64,' + figure1
  health_evaluation_figure_2.value = 'data:image/png;base64,' + figure2
  health_evaluation_figure_3.value = 'data:image/png;base64,' + figure3

  // const imgElement1 = document.getElementById('health_evaluation_figure_1'); 
  // const imgElement2 = document.getElementById('health_evaluation_figure_2');  
  // const imgElement3 = document.getElementById('health_evaluation_figure_3');   
  // imgElement1.src = `data:image/png;base64,${figure1}`; 
  // imgElement2.src = `data:image/png;base64,${figure2}`; 
  // imgElement3.src = `data:image/png;base64,${figure3}`; 

}
const health_evaluation_hide = () => {
  display_health_evaluation.value = false

}


// 特征提取结果展示
const display_feature_extraction = ref(false)
const transformedData = ref([])
const columns = ref([])

const feature_extraction_display = (results_object) => {
  // console.log('提取到的特征： ', results_object)
  display_feature_extraction.value = true
  let features_with_name = Object.assign({}, results_object.features_with_name)
  let features_name = features_with_name.features_name.slice()
  let features_group_by_sensor = Object.assign(features_with_name.features_extracted_group_by_sensor)
  let datas = []        // 表格中每一行的数据
  features_name.unshift('传感器')  // 表格的列名
  for (const sensor in features_group_by_sensor) {
    let features_of_sensor = features_group_by_sensor[sensor].slice()
    features_of_sensor.unshift(sensor)
    // console.log('features_of_sensor: ', features_of_sensor)
    datas.push(features_of_sensor)
    // features_of_sensor.splice(0, 1)
  }
  // console.log('features_name: ', features_name)
  // console.log('datas: ', datas)
  // let i = { prop: '', label: '', width: '' }
  columns.value.length = 0
  features_name.forEach(element => {
    // console.log('element: ', element)
    columns.value.push({ prop: element, label: element, width: 180 })
  });

  // console.log('columns: ', columns)
  // 转换数据为对象数组  
  transformedData.value = datas.map((row, index) => {
    const obj = {};
    columns.value.forEach((column, colIndex) => {
      obj[column.prop] = row[colIndex];
    });
    return obj;
  });
 
}


// 特征选择结果可视化
const display_feature_selection = ref(false)
const feature_selection_figure = ref('')
const features_selected = ref('')

const features_selection_display = (results_object) => {
  display_feature_selection.value = true

  let figure1 = results_object.figure_Base64
  features_selected.value = results_object.selected_features.join('、')
  feature_selection_figure.value = 'data:image/png;base64,' + figure1

}


// 故障诊断结果展示
const display_fault_diagnosis = ref(false)
const fault_diagnosis = ref('')
const fault_diagnosis_figure = ref('')

const fault_diagnosis_display = (results_object) => {
  display_fault_diagnosis.value = true

  let figure1 = results_object.figure_Base64
  let diagnosis_result = results_object.diagnosis_result
  if (diagnosis_result == 0) {
    fault_diagnosis.value = '无故障'
  } else {
    fault_diagnosis.value = '存在故障'
  }
  fault_diagnosis_figure.value = 'data:image/png;base64,' + figure1

}


// 故障预测结果展示
const display_fault_regression = ref(false)
const have_fault = ref(0)
const fault_regression = ref('')
const time_to_fault = ref('')
const fault_regression_figure = ref('')

const fault_regression_display = (results_object) => {
  display_fault_regression.value = true

  let figure1 = results_object.figure_Base64
  fault_regression_figure.value = 'data:image/png;base64,' + figure1
  // let fault_time = results_object.time_to_fault

  if (results_object.time_to_fault == 0) {
    have_fault.value = 1
    fault_regression.value = '已经出现故障'
  } else {
    have_fault.value = 0
    fault_regression.value = '还未出现故障'
    time_to_fault.value = results_object.time_to_fault_str
  }

}

// 插值处理可视化
const display_interpolation = ref(false)
const interpolation_figure = ref('')

const interpolation_display = (results_object) => {
  display_interpolation.value = true

  let figure1 = results_object.figure_Base64
  interpolation_figure.value = 'data:image/png;base64,' + figure1
}

// 无量纲化可视化
const display_normalization = ref(false)
const normalization_formdata_raw = ref([])
const normalization_formdata_scaled = ref([])
const normalization_columns = ref([])

const transform_data_to_formdata = (features_with_name: any, columns: any, formdata: any) => {

  let features_name = features_with_name.features_name.slice()
  let features_group_by_sensor = Object.assign(features_with_name.features_extracted_group_by_sensor)
  let datas = []        // 表格中每一行的数据
  features_name.unshift('传感器')  // 表格的列名
  for (const sensor in features_group_by_sensor) {
    let features_of_sensor = features_group_by_sensor[sensor].slice()
    features_of_sensor.unshift(sensor)
    // console.log('features_of_sensor: ', features_of_sensor)
    datas.push(features_of_sensor)
    // features_of_sensor.splice(0, 1)
  }
  // console.log('features_name: ', features_name)
  // console.log('datas: ', datas)
  // let i = { prop: '', label: '', width: '' }
  columns.value.length = 0
  features_name.forEach(element => {
    // console.log('element: ', element)
    columns.value.push({ prop: element, label: element, width: 180 })
  });

  // console.log('columns: ', columns)
  // 转换数据为对象数组  
  formdata.value = datas.map((row, index) => {
    const obj = {};
    columns.value.forEach((column, colIndex) => {
      obj[column.prop] = row[colIndex];
    });
    return obj;
  });
}

const normalization_display = (results_object: any) => {
  display_normalization.value = true

  let raw_data = Object.assign({}, results_object.raw_data)
  let scaled_data = Object.assign({}, results_object.scaled_data)

  transform_data_to_formdata(raw_data, normalization_columns, normalization_formdata_raw)
  transform_data_to_formdata(scaled_data, normalization_columns, normalization_formdata_scaled)
  console.log('normalization_formdata_raw: ', normalization_formdata_raw)
  console.log('normalization_formdata_scaled: ', normalization_formdata_scaled)
  console.log('normalization_columns: ', normalization_columns)

}

// 小波降噪可视化
const display_denoise = ref(false)
const denoise_figure = ref('')

const denoise_display = (results_object) => {
  display_denoise.value = true
  denoise_figure.value = 'data:image/png;base64,' + results_object.figure_Base64
}

// 清除可视化区域
const results_view_clear = () => {
  showPlainIntroduction.value = false  // 清除算法介绍
  showStatusMessage.value = false      // 清除程序运行状态
  result_show.value = false         // 清除可视化区域元素
  show1.value = true
  loading.value = true
  isshow.value = false
  // 清除所有结果可视化
  display_health_evaluation.value = false
  display_feature_extraction.value = false
  display_feature_selection.value = false
  display_fault_diagnosis.value = false
  display_fault_regression.value = false
  display_interpolation.value = false
  display_normalization.value = false
  display_denoise.value = false
}




// 做一个缓冲，以免用户快速点击致使后端崩溃
const refreshPardon = ref('http://127.0.0.1:7860')

// 根据节点信息发送结果展示请求
const resultShow = (item) => {
  // showMenu.value = false
  // isshow.value = false
  // isshow.value = true
  // display_module.label = item.label
  // console.log(display_module[label])
  if (done.value) {

    if (missionComplete.value) {
      if (item.label != '层次分析模糊综合评估' && item.label != '特征提取' && item.label != '特征选择' && item.label != '故障诊断'
        && item.label != '趋势预测' && item.label != '特征提取' && item.label != '插值处理' && item.label != '无量纲化' && item.label != '小波变换'
      ) {
        showPlainIntroduction.value = false
        showStatusMessage.value = false
        show1.value = true
        loading.value = true
        result_show.value = false
        isshow.value = false
        setTimeout(function () {
          isshow.value = true
          show1.value = false
          loading.value = false
        }, 2500); // 5000毫秒等于5秒
        let moduleName = item.label
        let url = 'http://127.0.0.1:8000/homepage?display=' + moduleName
        axios.request({
          method: 'GET',
          url: url,
        });
        setTimeout(function () {
          // 为 iframe 的 src 属性添加一个查询参数，比如当前的时间戳，以强制刷新
          var iframe = document.getElementById('my_gradio_app');
          var currentSrc = iframe.src;
          var newSrc = currentSrc.split('?')[0]; // 移除旧的查询参数
          iframe.src = newSrc + '?updated=' + new Date().getTime();
        }, 2400);
      } else {
        results_view_clear()
        result_show.value = true
        if (item.label == '层次分析模糊综合评估') {
          let results_to_show = response_results.层次分析模糊综合评估
          health_evaluation_display(results_to_show)
        } else if (item.label == '特征提取') {
          let results_to_show = response_results.特征提取
          feature_extraction_display(results_to_show)
        } else if (item.label == '特征选择') {
          let results_to_show = response_results.特征选择
          features_selection_display(results_to_show)
        } else if (item.label == '故障诊断') {
          let results_to_show = response_results.故障诊断
          fault_diagnosis_display(results_to_show)
        } else if (item.label == '趋势预测') {
          let results_to_show = response_results.趋势预测
          fault_regression_display(results_to_show)
        } else if (item.label == '插值处理') {
          let results_to_show = response_results.插值处理
          interpolation_display(results_to_show)
        } else if (item.label == '无量纲化') {
          let results_to_show = response_results.无量纲化
          normalization_display(results_to_show)
        } else if (item.label == '小波变换') {
          let results_to_show = response_results.小波变换
          denoise_display(results_to_show)
        }
        else {
          ElMessage({
            message: '无效的算法模块',
            type: 'error'
          })
        }
      }



    }
  } else {


  }
}




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
  data_drawer.value = false
  models_drawer.value = true
  let url = 'http://127.0.0.1:8000/user_fetch_models/'
  api.request({
    method: 'GET',
    url: url
  }).then((response) => {
    let modelsInfo = response.data
    fetchedModelsInfo.value.length = 0
    for (let item of modelsInfo) {
      fetchedModelsInfo.value.push(item)
    }
  })
}

// 从后端获取到的历史模型的信息
const fetchedModelsInfo = ref([])


// 复用历史模型，不需要进行模型检查等操作
let model_has_been_saved = false

// 点击历史模型表格中使用按钮触发复现历史模型
const use_model = (row) => {

  if (nodeList.value.length != 0) {
    nodeList.value.length = 0
  }
  handleclear()
  updateStatus('当前模型已保存')
  model_has_been_saved = true
  canStartProcess.value = false
  let objects = JSON.parse(row.model_info)
  let node_list = objects.nodeList         // 模型节点信息   
  let connection = objects.connection      // 模型连线信息
  // for (let i = 0; i < node_list.length; i++){

  //   node_list[i].nodeContainerStyle.left -= 490
  //   node_list[i].nodeContainerStyle.top -= 120
  //   node_list[i].nodeContainerStyle.left += 'px'
  //   node_list[i].nodeContainerStyle.top += 'px'

  // }
  // 恢复节点
  for (let node of node_list) {

    nodeList.value.push(node)

    if (node.label == '特征提取') {
      features.value.length = 0
      let params = node.parameters[node.use_algorithm]
      for (let [key, value] of Object.entries(params)) {
        if (value) {
          features.value.push(key)
        }
      }
    }
  }
  let id_to_label_list = { 'node_id': [], 'node_label': [] }
  // 初始化每个节点的可连接状态
  for (let node of nodeList.value) {
    let nodeId = node.id
    id_to_label_list.node_id.push(nodeId)
    id_to_label_list.node_label.push(node.label)
    nextTick(() => {
      // plumbIns.draggable(nodeId, { containment: "efContainer" })
      if (node.id === '2.2') {
        plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)
        return
      }
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions)
      // plumbIns.addEndpoint(nodeId, deff.jsplumbTargetOptions)
      if (node.id === '1') {
        return
      }
      plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)
    })
  }

  // 恢复连线
  let connection_list = []
  let connection2 = []
  let node_num = connection.length
  console.log(connection)
  for (let i = 0; i < node_num; i++) {
    let label = connection[i]
    for (let j = 0; j < node_num; j++) {
      if (id_to_label_list.node_label[j] === label) {
        connection2[i] = id_to_label_list.node_id[j]
        break
      }
    }
  }
  console.log(connection2)
  saveModelSetting(false, connection)
  content_json.schedule = connection
  console.log('conten_json3: ', content_json)
  modelSetup.value = true
  if (node_num == 1) {
    connection_list = []
  } else {
    for (let i = 0; i < node_num - 1; i++) {
      connection_list.push({ 'soruce_id': connection2[i], 'target_id': connection2[i + 1] })
    }
    nextTick(() => {
      for (let line of connection_list) {
        plumbIns.connect({
          source: document.getElementById(line.soruce_id),
          target: document.getElementById(line.target_id)
        })
      }
    })
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

}

// 删除模型操作
let index = 0
let row = 0
const delete_model_confirm_visible = ref(false)

const delete_model = (index_in, row_in) => {
  index = index_in
  row = row_in
  delete_model_confirm_visible.value = true
}

const delete_model_confirm = () => {

  // 发送删除请求到后端，row 是要删除的数据行
  let url = 'http://127.0.0.1:8000/user_delete_model/?row_id=' + row.id
  api.request({
    method: 'GET',
    url: url
  }).then((response) => {
    if (response.data.message == 'deleteSuccessful') {
      if (index !== -1) {
        // 删除前端表中数据
        fetchedModelsInfo.value.splice(index, 1)
        delete_model_confirm_visible.value = false
      }
    }
  }).catch(error => {
    // 处理错误，例如显示一个错误消息  
    console.error(error);
  });
}

// 查看模型信息
const model_name = ref('')
const model_algorithms = ref([])
const model_params = ref([])  // {'模块名': xx, '算法': xx, '参数': xx}

const show_model_info = (row) => {
  let objects = JSON.parse(row.model_info)
  let node_list = objects.nodeList         // 模型节点信息   
  let connection = objects.connection     // 模型连接顺序

  model_name.value = row.model_name
  model_algorithms.value = connection
  model_params.value.length = 0
  node_list.forEach(element => {
    let item = { '模块名': '', '算法': '' }
    item.模块名 = element.label
    item.算法 = labels_for_algorithms[element.use_algorithm]
    model_params.value.push(item)
  });
}

const showStatusMessage = ref(false)
const status_message_to_show = ref('')

const status_message = {
  'success': '## 程序已经运行完毕，请点击相应的算法模块查看对应结果！',
  'shutdown': '## 程序运行终止，点击重置模型重新建立模型'
}

//按钮失效控制

// 控制是否可以修改模型
const to_rectify_model = ref(false)

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
        message: '完成建模',
        type: 'success'
      })
      canCheckModel.value = false
      modelSetup.value = true     // 不能删除建模区模块
      done.value = true     // 不能拖动模块
      to_rectify_model.value = true
      updateStatus('模型建立完成但未通过检查')
      return
    }
    if (linkedList.length() != nodeList.value.length) {
      ElMessage({
        message: '请确保图中所有模块均已建立连接，且没有多余的模块',
        type: 'warning'
      })
      return
    }
  }

  ElMessage({
    message: '完成建模',
    type: 'success'
  })
  modelSetup.value = true     // 不能删除建模区模块
  done.value = true     // 不能拖动模块
  to_rectify_model.value = true
  canCheckModel.value = false
  updateStatus('模型建立完成但未通过检查')
}

// 修改模型
const rectify_model = () => {
  canCheckModel.value = true
  canSaveModel.value = true
  canStartProcess.value = true
  canShutdown.value = true
  modelSetup.value = false     // 可以删除建模区模块
  done.value = false     // 可以拖动模块
  to_rectify_model.value = false
  ElMessage({
    showClose: true,
    message: '进行模型修改, 完成修改后请再次点击完成建模',
    type: 'info'
  })
  updateStatus('正在修改模型')
}

//检查模型
const checkModeling = () => {
  if (nodeList.value.length == 0 && !model_has_been_saved) {
    canCheckModel.value = true
  }
}

//保存模型
const saveModeling = () => {
  if (nodeList.value.length == 0 || model_has_been_saved) {
    canSaveModel.value = true
  }
}

//开始
const startModeling = () => {
  if (nodeList.value.length == 0) {
    canStartProcess.value = true
  }
}


// 建模状态更新
function updateStatus(status) {
  var indicator = document.getElementById('statusIndicator');
  indicator.textContent = status; // 更新文本  
  indicator.classList.remove('error', 'success', 'saved', 'rectify'); // 移除之前的状态类  
  switch (status) {
    case '未建立模型':
      // 默认样式，或者设置为特定类  
      break;
    case '模型建立完成但未通过检查':
      indicator.classList.add('error');
      break;
    case '模型建立并已通过模型检查':
      indicator.classList.add('success');
      break;
    case '当前模型已保存':
      indicator.classList.add('saved');
    case '正在修改模型':
      indicator.classList.add('rectify')
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
  baseURL: 'http://127.0.0.1:8000',
});


// 拦截请求并将登录时从服务器获得的token添加到Authorization头部  
api.interceptors.request.use(function (config) {
  // 从localStorage获取JWT  
  let token = window.localStorage.getItem('jwt');
  console.log('the token is: ', token)

  // 将JWT添加到请求的Authorization头部  
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;

}, function (error) {
  // 请求错误处理  
  return Promise.reject(error);
})


const fetchedDataFiles = ref<Object[]>([])

// 用户目前选择的数据文件
const using_datafile = ref('')

const delete_dataset_confirm_visible = ref(false)
let row_dataset: any = null
let index_dataset: any = null
// 用户删除历史数据
const delete_dataset = (index_in: any, row_in: any) => {
  index_dataset = index_in
  row_dataset = row_in
  delete_dataset_confirm_visible.value = true
}

const delete_dataset_confirm = () => {

  api.get('/delete_datafile?filename=' + row_dataset.dataset_name)
    .then((response: any) => {
      if (response.data.code == 200){
        // 删除前端表中数据
        fetchedDataFiles.value.splice(index_dataset, 1)
        delete_dataset_confirm_visible.value = false
        ElMessage({
          message: '文件删除成功',
          type: 'success'
        })
      }else if(response.data.code == 400){
        ElMessage({
          message: '删除失败: ' + response.data.message,
          type: 'error'
        })
      }
     
    })
    .catch(error => {
      console.log('delete_datafile_error: ', error)
      ElMessage({
        message: '删除失败',
        type: 'error'
      })
    })
}


const loading_data = ref(false)
// 用户选择历史数据
const use_dataset = (row_in: any) => {
  loading_data.value = true
  setTimeout(() => {
    loading_data.value = false
    using_datafile.value = row_in.dataset_name
    ElMessage({
      message: '数据加载成功',
      type: 'success'
    })
}, 1000)
  
}

const handleSwitchDrawer = (fetchData: any[]) => {

  models_drawer.value = false;

  fetchedDataFiles.value = []
  // fetchData.forEach(element => {
  //   fetchedDataFiles.value.push(element)
  // });
  for (let item of fetchData){
    fetchedDataFiles.value.push(item)
  }
  nextTick(() => {
    console.log('nextTick')
    console.log('fetchedDataFiles: ', fetchedDataFiles.value)
  })

  data_drawer.value = true
  
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

ul>li {
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
  top: 5px
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

.node-info-label:hover+.node-drag {
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
  background-image: url('../assets/可视化建模.png');
  background-position: center;
  background-size: contain;
  /* height: 50vh; */
  background-repeat: no-repeat;
}

.clickable:hover {
  cursor: pointer;
  color: #007BFF;
}
</style>
