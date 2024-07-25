import {defineStore} from 'pinia'
import {ref} from 'vue'

function initState () {
    return{
        isCollapse: false,
        currentMenu: {
            key: '',
            label: '',
            icon: '',
            path: ''
        }
    }
}
export const useAllDataStore = defineStore('AllData', () => {

    const state = ref(initState())

    return { state }
})